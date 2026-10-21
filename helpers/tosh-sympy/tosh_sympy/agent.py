# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The engine's math agent. llama-server started with --mcp-agent hands every
/v1/chat/completions request that brings no tools to the "run" tool of this server, which
answers it with the math tools and the rules in policy.py, talking to the engine over HTTP:
raw chat completions to generate, /tools to compute.

Without the math tools on the engine, the request is answered by the model as it is.
"""

import http.client
import json
import os
import re
import socket
import threading
import time
import urllib.parse
import uuid

from . import policy

TRUST_KEY = os.environ.get("TOSH_TRUST_KEY", "")
# MCP servers an administrator approved for the agent; the engine names their tools "<server>_<tool>"
APPROVED = tuple(name + "_" for name in os.environ.get("TOSH_AGENT_SERVERS", "").split(",") if name)
MAX_ROUNDS = 10
# what a client may set on a request that the agent passes on to each round
_GENERATION = ("temperature", "top_p", "top_k", "min_p", "seed", "repeat_penalty", "presence_penalty",
               "frequency_penalty", "max_tokens", "max_completion_tokens", "stop", "model", "chat_template_kwargs",
               "repeat_last_n", "typ_p", "typical_p", "dynatemp_range", "dynatemp_exponent", "xtc_probability",
               "xtc_threshold", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
               "samplers", "backend_sampling", "thinking_budget_tokens", "reasoning_control", "cache_prompt", "id_slot")

DEFINITION = {
    "name": "run",
    "description": "Answers an OpenAI chat completions request with the math tools. Called by the engine.",
    "inputSchema": {"type": "object", "properties": {"request": {"type": "object"}, "base_url": {"type": "string"},
                                                    "api_key": {"type": "string"}, "agent_key": {"type": "string"}},
                    "required": ["request", "base_url"], "additionalProperties": False},
    "annotations": {"readOnlyHint": True},
}


class Invalid(Exception):
    pass


class Cancelled(Exception):
    pass


# the engine's own words for a request that does not fit its context, and for a GPU that failed while working
_CONTEXT = re.compile(r"exceeds? the available context size|context size has been exceeded|exceed_context_size", re.I)
_GPU = re.compile(r"compute error|error state|kIOAccel|command buffer|failed to (?:process|encode) (?:mtmd|image)|"
                  r"\bmetal\b|\bgpu\b", re.I)
FAILURE_TEXT = {
    "context": {"en": "The request needs {tokens} tokens but the model's context holds {context}, so it was not answered. "
                      "Raise the context size in the server settings, send less, or start a new conversation.",
                "es": "La petición necesita {tokens} tokens pero el contexto del modelo admite {context}, así que no se "
                      "respondió. Sube el tamaño de contexto en los ajustes del servidor, envía menos o empieza una "
                      "conversación nueva."},
    "gpu": {"en": "The engine failed on the GPU while processing this request ({detail}), so the request did not finish "
                  "and nothing was answered from it. The engine has to be restarted to recover; the conversation is kept, "
                  "and the message can be sent again once it is back.",
            "es": "El motor falló en la GPU mientras procesaba esta petición ({detail}), así que la petición no terminó y no "
                  "se respondió nada a partir de ella. Hay que reiniciar el motor para recuperarlo; la conversación se "
                  "conserva y el mensaje se puede volver a enviar cuando esté de nuevo listo."},
    "server": {"en": "The engine failed while processing this request ({detail}), so it did not finish.",
               "es": "El motor falló mientras procesaba esta petición ({detail}), así que no terminó."},
}
CUT = {"en": "_The answer was cut at the limit of tokens per answer, so it is incomplete._",
       "es": "_La respuesta se cortó en el límite de tokens por respuesta, así que está incompleta._"}


class EngineFailure(RuntimeError):
    """The engine could not run a pass: the request does not fit its context, its GPU failed, or it answered with an
    error of its own. The turn ends saying so, never with an answer that hides it."""

    STATUS = {"context": 400, "gpu": 503, "server": 502}
    TYPE = {"context": "exceed_context_size_error", "gpu": "engine_gpu_error", "server": "engine_error"}

    def __init__(self, kind, detail, tokens=None, context=None):
        super().__init__(detail)
        self.kind, self.detail, self.tokens, self.context = kind, detail, tokens, context

    def answer(self, lang):
        text = FAILURE_TEXT[self.kind].get(lang, FAILURE_TEXT[self.kind]["en"])
        error = {"message": text.format(detail=self.detail[:200], tokens=self.tokens or "?", context=self.context or "?"),
                 "type": self.TYPE[self.kind], "code": self.STATUS[self.kind], "engine_message": self.detail[:300]}
        if self.tokens and self.context:
            error.update(n_prompt_tokens=self.tokens, n_ctx=self.context)
        return {"status": self.STATUS[self.kind], "error": error}


def failure(reply):
    """The EngineFailure an error reply of the engine states, or None for a reply without one."""
    error = reply.get("error") if isinstance(reply, dict) else None
    if error is None:
        return None
    if not isinstance(error, dict):
        error = {"message": str(error)}
    message = str(error.get("message") or error.get("type") or "the engine gave no answer")
    if _CONTEXT.search(message) or _CONTEXT.search(str(error.get("type") or "")):
        return EngineFailure("context", message, error.get("n_prompt_tokens"), error.get("n_ctx"))
    return EngineFailure("gpu" if _GPU.search(message) else "server", message)


def _raise_failure(reply):
    """A side pass that hit a context too small or a failed GPU ends the turn; any other error leaves it unanswered."""
    found = failure(reply)
    if found and found.kind != "server":
        raise found


def _gpu_checked(reply):
    """The reply, unless it says the GPU failed: no later pass of the turn can run, so the turn ends on it."""
    found = failure(reply)
    if found and found.kind == "gpu":
        raise found
    return reply


class Engine:
    def __init__(self, base_url, api_key, agent_key=""):
        parsed = urllib.parse.urlparse(base_url)
        # an IPv6 host comes in brackets with its zone's % as %25, and connects without them
        self.host, self.port = urllib.parse.unquote(parsed.hostname or "127.0.0.1"), parsed.port or 80
        self.api_key = api_key or ""
        # what /tools asks before it lists or runs the tools of a server only the agent may use
        self.agent_key = agent_key or ""
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._connections = set()

    def cancel(self):
        """Stops the requests in flight: the engine sees each connection close and stops generating."""
        with self._lock:
            self.cancelled.set()
            connections = list(self._connections)
        for connection in connections:
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _send(self, method, path, body=None, timeout=3600, on_line=None):
        """The engine's JSON reply. With on_line, a 200 reply is read as server-sent events, each line to
        on_line, and None is returned."""
        if self.cancelled.is_set():
            raise Cancelled()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        # rounds of the agent are raw inference, never the agent again
        headers = {"Content-Type": "application/json", "X-Tosh-Agent": "off"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if self.agent_key:
            headers["X-Tosh-Agent-Key"] = self.agent_key
        # connected before it is shared, so a cancellation always finds a socket to close
        connection.connect()
        with self._lock:
            if self.cancelled.is_set():
                connection.close()
                raise Cancelled()
            self._connections.add(connection)
        try:
            connection.request(method, path, None if body is None else json.dumps(body), headers)
            response = connection.getresponse()
            if on_line is not None and response.status == 200:
                for line in response:
                    on_line(line.decode("utf-8", "replace").rstrip("\r\n"))
                data = None
            else:
                data = response.read()
        except (OSError, http.client.HTTPException):
            if self.cancelled.is_set():
                raise Cancelled()
            raise
        finally:
            with self._lock:
                self._connections.discard(connection)
            connection.close()
        if self.cancelled.is_set():
            raise Cancelled()
        if on_line is not None and data is None:
            return None
        try:
            return json.loads(data or b"null")
        except ValueError:
            return {"error": {"message": "the engine answered something that is not JSON"}}

    def complete(self, body):
        return _gpu_checked(self._send("POST", "/v1/chat/completions", dict(body, stream=False)))

    def stream(self, body, on_delta):
        """A completion generated as a stream: on_delta gets each piece of text as (field, text), with field
        "content" or "reasoning_content". Returns the reply in the shape of complete()'s."""
        message, usage, failure = {"role": "assistant", "content": ""}, {}, []

        def on_line(line):
            if not line.startswith("data: ") or line == "data: [DONE]":
                return
            try:
                chunk = json.loads(line[6:])
            except ValueError:
                return
            if not isinstance(chunk, dict):
                return
            if "error" in chunk:
                failure.append(chunk)
                return
            if isinstance(chunk.get("usage"), dict):
                usage.update(chunk["usage"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(delta, dict):
                    _merge(message, delta, on_delta)

        reply = self._send("POST", "/v1/chat/completions",
                           dict(body, stream=True, stream_options={"include_usage": True}), on_line=on_line)
        if reply is not None:
            return _gpu_checked(reply)
        if failure:
            return _gpu_checked(failure[0])
        return {"choices": [{"message": message}], "usage": usage}

    def tools(self):
        listed = self._send("GET", "/tools", timeout=60)
        return [t for t in listed if isinstance(t, dict) and (policy.is_math(t.get("tool")) or approved(t.get("tool")))] \
            if isinstance(listed, list) else []

    def call(self, name, params, timeout=120):
        return self.call_checked(name, params, timeout)[0]

    def call_checked(self, name, params, timeout=120):
        """The tool's text and whether it ran without an error."""
        reply = self._send("POST", "/tools", {"tool": name, "params": params}, timeout=timeout)
        if isinstance(reply, dict):
            if isinstance(reply.get("plain_text_response"), str):
                return reply["plain_text_response"], True
            if isinstance(reply.get("error"), str):
                return reply["error"], False
        return json.dumps(reply), False


def _merge(message, delta, on_delta):
    for field in ("content", "reasoning_content"):
        text = delta.get(field)
        if isinstance(text, str) and text:
            message[field] = message.get(field, "") + text
            on_delta(field, text)
    for fragment in delta.get("tool_calls") or []:
        if not isinstance(fragment, dict):
            continue
        calls = message.setdefault("tool_calls", [])
        index = fragment.get("index") if isinstance(fragment.get("index"), int) else len(calls)
        while len(calls) <= index:
            calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if fragment.get("id"):
            calls[index]["id"] = fragment["id"]
        function = fragment.get("function") if isinstance(fragment.get("function"), dict) else {}
        for key in ("name", "arguments"):
            if isinstance(function.get(key), str):
                calls[index]["function"][key] += function[key]


def approved(name):
    return bool(APPROVED) and isinstance(name, str) and name.startswith(APPROVED) and not policy.is_math(name)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _media(request):
    """Whether the request's last message carries images, audio or video, which this agent cannot read."""
    messages = request.get("messages")
    content = messages[-1].get("content") if isinstance(messages, list) and messages and isinstance(messages[-1], dict) else None
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") not in (None, "text") for p in content)


def _parse(request):
    """(system, history, question) from a chat request. Tool messages are refused: the tools run
    here, and a result sent by a client could otherwise pass for one Tosh computed."""
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise Invalid("messages must be a non-empty array")
    system, turns = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise Invalid("each message must be an object")
        role = message.get("role")
        if role in ("tool", "function") or message.get("tool_calls") or message.get("function_call"):
            raise Invalid("tool calls and tool results are not accepted: the tools run inside Tosh")
        text = _text(message.get("content"))
        if role in ("system", "developer"):
            system.append(text)
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": text})
        else:
            raise Invalid(f"unknown role {str(role)[:20]!r}")
    if not turns or turns[-1]["role"] != "user" or not turns[-1]["content"].strip():
        raise Invalid("the last message must come from the user")
    return "\n\n".join(system), turns[:-1], turns[-1]["content"]


class Turn:
    def __init__(self, engine, request, emit=None):
        self.engine = engine
        self.request = request
        self.emit = emit or (lambda event: None)
        self.system, self.history, self.question = _parse(request)
        self.lang = policy.language([t["content"] for t in self.history if t["role"] == "user"] + [self.question])
        self.sources = [t["content"] for t in self.history if t["role"] == "user"] + [self.question]
        self.messages = []          # this turn's rounds, as the model reads them
        self.calls = []             # every call of the turn and how it ended
        self.listed = set()         # tool names the engine offered this turn
        self.definitions = {}       # and their definitions
        self.guard = policy.Guard()
        self.passes = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.intent = policy.NO_MATH
        # why the last model pass stopped: "length" when it ran out of tokens
        self.finish = None

    def _body(self, tools, choice, note):
        body = {k: self.request[k] for k in _GENERATION if k in self.request}
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        if body["chat_template_kwargs"].get("enable_thinking") is False:
            body.setdefault("thinking_budget_tokens", 0)
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + self.history
        messages = messages + [{"role": "user", "content": self.question}] + self.messages
        if note:
            messages = messages + [{"role": "user", "content": note + policy.language_note(_text(self.question))}]
        if body["chat_template_kwargs"].get("enable_thinking") is False:
            # Qwen3 templates that ignore enable_thinking still honour the switch in the text, as the app's chat sends it
            last = max(i for i, m in enumerate(messages) if m["role"] == "user")
            messages[last] = dict(messages[last], content=messages[last]["content"] + "\n/no_think")
        body["messages"] = messages
        if tools:
            body["tools"] = tools
            body["tool_choice"] = choice
        return body

    def _generate(self, body, live=False):
        """One model pass. A live pass streams its text to the client as it is generated, for a round
        whose answer goes out unchecked; if it ends in tool calls, the client is told to drop that text."""
        self.passes += 1
        self.emit({"type": "pass", "n": self.passes, "state": "start", "live": live})
        shown = False
        # text of a pass that may still end in tool calls is a draft: a client that only reads content gets the answer
        drafting = bool(body.get("tools")) and body.get("tool_choice") != "none"

        def on_delta(field, text):
            nonlocal shown
            shown = shown or field == "content"
            self.emit({"type": "draft" if drafting and field == "content" else "delta", field: text})

        reply = self.engine.stream(body, on_delta) if live else self.engine.complete(body)
        if not isinstance(reply, dict) or not reply.get("choices"):
            error = reply.get("error") if isinstance(reply, dict) and isinstance(reply.get("error"), dict) else {}
            found = failure(reply)
            if found and found.kind == "context":
                raise found
            # a request the engine refuses (no model in router mode...) is the client's to fix
            if error.get("type") == "invalid_request_error" or error.get("code") == 400:
                raise Invalid(str(error.get("message") or "the engine refused the request"))
            raise found or RuntimeError("the engine gave no answer")
        self.finish = reply["choices"][0].get("finish_reason")
        usage = reply.get("usage") or {}
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.emit({"type": "pass", "n": self.passes, "state": "end", "prompt_tokens": usage.get("prompt_tokens", 0),
                   "completion_tokens": usage.get("completion_tokens", 0)})
        message = reply["choices"][0].get("message") or {}
        if shown and message.get("tool_calls"):
            self.emit({"type": "retract", "n": self.passes})
        return message

    def _rounds(self):
        """The client's own limit on model passes ("tosh": {"max_rounds": n}), as the app's agent turns setting."""
        options = self.request.get("tosh")
        value = options.get("max_rounds") if isinstance(options, dict) else None
        return max(1, min(100, value)) if isinstance(value, int) and not isinstance(value, bool) else MAX_ROUNDS

    def _routed(self, body):
        """A side question to the model goes to the same model as the turn, which a router needs."""
        if self.request.get("model"):
            body["model"] = self.request["model"]
        return body

    def _intent(self):
        decided = policy.classify(self.question, [t["content"] for t in self.history if t["role"] == "user"])
        if decided:
            return decided, False
        reply = self.engine.complete(self._routed({
            "messages": [{"role": "system", "content": policy.INTENT_INSTRUCTIONS},
                         {"role": "user", "content": self.question[:4000] + " /no_think"}],
            "max_tokens": 6, "temperature": 0, "grammar": policy.INTENT_GRAMMAR, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}))
        _raise_failure(reply)
        try:
            word = reply["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            word = ""
        if word in policy.INTENTS:
            return word, True
        return (policy.COMPUTATIONAL if policy.evidence(self.question)["structure"] else policy.CONCEPTUAL), True

    def _source(self):
        """The user's words and what the tools returned this turn: refused and failed calls stay out."""
        context = [t["content"] for t in self.history if t["role"] == "user"]
        context += [c["result"] for c in self.calls
                    if policy.succeeded(c) or (approved(c["name"]) and c["state"] == "completed")]
        return {"request": self.question, "context": "\n".join(context)[-6000:]}

    def _review(self, reply, operation):
        lines = reply.get("interpreted_input") or []
        # a number an approved server returned this turn is as given as the user's own
        returned = [c["result"][:1500] for c in self.calls if approved(c["name"]) and c["state"] == "completed"]
        request = self.question + ("\n\nRETURNED BY THE TOOLS FOR THIS REQUEST:\n" + "\n".join(returned) if returned else "")
        answer = self.engine.complete(self._routed({
            "messages": [{"role": "system", "content": policy.REVIEW_INSTRUCTIONS},
                         {"role": "user", "content": f"REQUEST:\n{request}\n\nCALL: {operation or ''}\n"
                                                     + "\n".join(lines) + " /no_think"}],
            "max_tokens": 4, "temperature": 0, "grammar": policy.REVIEW_GRAMMAR, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}))
        _raise_failure(answer)
        try:
            return answer["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    def _run_call(self, call):
        name = call.get("function", {}).get("name", "")
        raw = call.get("function", {}).get("arguments") or "{}"
        entry = {"id": call.get("id") or uuid.uuid4().hex, "name": name, "arguments": raw, "state": "failed",
                 "result": "", "reply": None}
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            arguments = None
        if not isinstance(arguments, dict):
            entry["result"] = json.dumps({"success": False, "error": {"code": "invalid_arguments",
                                                                       "message": "the arguments are not a JSON object"}})
        elif approved(name) and name in self.listed:
            # an approved server's tool gets the model's arguments as they are; the math checks are not its
            plain = {k: v for k, v in arguments.items() if not str(k).startswith("_")}
            ungiven = self._ungiven(name, plain)
            if ungiven:
                entry["result"] = json.dumps({"success": False, "error": {
                    "code": "needs_user_input",
                    "message": "the request never gave " + ", ".join(ungiven) + "; ask the user with "
                               + policy.CLARIFY + " instead of guessing"}})
                entry["reply"] = json.loads(entry["result"])
            else:
                text, ok = self.engine.call_checked(name, plain, timeout=900)
                entry.update(result=text, state="completed" if ok else "failed")
        elif not policy.is_math(name):
            entry["result"] = json.dumps({"success": False, "error": {"code": "invalid_arguments",
                                                                       "message": f"no such tool: {name[:60]}"}})
        else:
            # the model's own "_" fields never reach a tool; the trusted ones are added here
            plain = {k: v for k, v in arguments.items() if not str(k).startswith("_")}
            params = dict(plain, _source=self._source(), _trust=TRUST_KEY)
            text = self.engine.call(name, params)
            reply = policy.reply_of(text)
            if policy.error_code(reply) == "needs_review" and self._review(reply, plain.get("operation")) == "consistent":
                text = self.engine.call(name, dict(params, _reviewed="consistent"))
                reply = policy.reply_of(text)
            entry.update(result=text, reply=reply, state="completed" if reply and reply.get("success") else "failed")
            if reply is not None and "timed_out" in reply and reply.get("success") is not True:
                entry["state"] = "completed"
        self.calls.append(entry)
        self.emit({"type": "tool_call", "pass": self.passes,
                   "call": dict(_describe(entry), id=entry["id"], state=entry["state"], result=entry["result"][:20000])})
        return entry

    def _clarification(self):
        """The question for the user after an approved server's tool lacked a value; None when the turn never reached
        one, and the math wording applies."""
        reached = [c for c in self.calls if approved(c["name"])]
        if not reached:
            return None
        wanted = []
        for c in reached:
            if (c.get("reply") or {}).get("error", {}).get("code") == "needs_user_input":
                try:
                    wanted += [k for k in json.loads(c["arguments"]) if k not in wanted]
                except (ValueError, TypeError, AttributeError):
                    pass
        named = ", ".join(f"`{k}`" for k in wanted)
        if self.lang == "es":
            return f"Para responder necesito que me indiques {named}." if named else "Para responder necesito más datos. ¿Puedes concretar la petición?"
        return f"To answer, I need you to tell me {named}." if named else "To answer, I need more details. Could you make the request more specific?"

    def _ungiven(self, name, arguments):
        """Numbers and single words an approved tool would get that neither the user nor an earlier result gave.
        Free text with spaces, booleans and the values the tool's schema lists are the model's to write."""
        schema = ((self.definitions.get(name) or {}).get("function") or {}).get("parameters") or {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        given = "\n".join([_text(t["content"]) for t in self.history if t["role"] == "user"] + [self.question]
                          + [c["result"] for c in self.calls if c["state"] == "completed"]).lower()
        numbers = {value for _, value, _, _ in policy.numbers(given)}
        ungiven = []
        for key, value in arguments.items():
            if isinstance(value, bool) or value in ((properties.get(key) or {}).get("enum") or []):
                continue
            if isinstance(value, (int, float)) and float(value) not in numbers:
                ungiven.append(f"{key}={value}")
            elif isinstance(value, str) and value.strip() and " " not in value.strip():
                word = value.strip().lower()
                # the parameter's own name or a <placeholder> is not a value either
                if word not in given or word == key.lower() or "<" in word or ">" in word:
                    ungiven.append(f"{key}={value.strip()!r}")
        return ungiven

    def run(self):
        listed = self.engine.tools()
        # an approved server's tool runs only if the engine listed it
        self.listed = {t.get("tool") for t in listed}
        self.definitions = {t.get("tool"): t.get("definition") for t in listed}
        if not listed:
            # no math tools on this engine: the model answers as it is
            message = self._generate(self._body(None, None, None), live=True)
            return message.get("content") or "", "answered"
        # the model leans to the first tool that fits: exact ones go before their numerical twins
        tools = [t["definition"] for t in sorted(listed, key=lambda t: not str(t.get("tool", "")).startswith("sympy_"))
                 if isinstance(t.get("definition"), dict)]
        # with an approved server the model may also ask for what its tools need, in any round
        clarify_anywhere = any(approved(t.get("tool")) for t in listed)
        self.intent, asked = self._intent()
        self.emit({"type": "intent", "intent": self.intent, "asked_model": asked})
        # a calculation must go through the math tools, so it only binds when the engine has them
        gate = policy.requires_tools(self.intent) and any(policy.is_math(t.get("tool")) for t in listed)
        required = gate
        note, finalizing, regrounded = None, False, False
        for round_number in range(self._rounds()):
            first = round_number == 0
            if finalizing:
                body = self._body(None, None, note)
            elif (first and gate) or self.guard.next == "math_only":
                body = self._body(tools + [policy.CLARIFY_TOOL], "required", note)
            else:
                standing = note or (policy.STANDING_NOTE if any(policy.is_math(c["name"]) for c in self.calls) else None)
                body = self._body(tools + ([policy.CLARIFY_TOOL] if clarify_anywhere else []), "auto", standing)
            note = None
            # nothing checks the answer of a turn that needs no tools until a math tool has run in it
            live = not (required or finalizing or any(policy.is_math(c["name"]) for c in self.calls))
            message = self._generate(body, live)
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []
            names = [c.get("function", {}).get("name", "") for c in calls]
            arguments = [c.get("function", {}).get("arguments", "") for c in calls]
            closing = None
            if not finalizing:
                closing = self.guard.closing(names, arguments, self.lang)
                if closing is None and policy.CLARIFY in names and ((first and gate) or clarify_anywhere):
                    closing = self._clarification() or \
                        policy.unresolved(policy._missing(arguments[names.index(policy.CLARIFY)]), self.lang)
            if calls and closing is None and not finalizing:
                entries = [self._run_call(c) for c in calls]
                for entry in entries:
                    self.guard.record(entry["name"], entry["reply"])
                maths = [e for e in entries if policy.is_math(e["name"])]
                kept = content if all(policy.succeeded(e) for e in maths) else ""
                self.messages.append({"role": "assistant", "content": kept, "tool_calls": [
                    {"id": e["id"], "type": "function", "function": {"name": e["name"], "arguments": _readable(e["arguments"])}}
                    for e in entries]})
                self.messages += [{"role": "tool", "tool_call_id": e["id"], "content": e["result"][:20000]} for e in entries]
                if self.guard.next == "stop":
                    if policy.ledger(self.calls):
                        finalizing, note = True, policy.final_note(policy.ledger(self.calls))
                        continue
                    return policy.unresolved(lang=self.lang), "unresolved"
                continue
            # what an approved server returned is a source the answer may quote, like the user's words
            sources = self.sources + [c["result"] for c in self.calls if approved(c["name"]) and c["state"] == "completed"]
            kind, value = policy.step(sources, self.calls, content, closing, finalizing, regrounded, required, self.lang)
            if kind == "keep":
                final = closing if closing is not None else content
                if closing is not None:
                    return final, "clarification_required" if policy.CLARIFY in names else "unresolved"
                return final, "answered"
            if kind == "replace":
                return value, "validated_results_only" if policy.is_safe_answer(value) else "unresolved"
            if kind == "again":
                regrounded, note = True, value
            else:
                finalizing, note = True, value
        results = policy.ledger(self.calls)
        return policy.safe_answer(results, [c for c in self.calls if policy.is_math(c["name"])], self.lang), "turn_limit"


def _readable(arguments):
    # the engine refuses a history holding a call it cannot parse, such as one the model cut short
    if isinstance(arguments, dict):
        return arguments
    try:
        return arguments if isinstance(json.loads(arguments), dict) else "{}"
    except (TypeError, ValueError):
        return "{}"


def _describe(call):
    reply = call.get("reply") or {}
    try:
        arguments = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
    except ValueError:
        arguments = call["arguments"]
    entry = {"tool": call["name"], "arguments": arguments,
             "status": "ok" if policy.succeeded(call) or (approved(call["name"]) and call["state"] == "completed")
             else policy.error_code(reply) or "execution_error"}
    if reply.get("interpreted_input"):
        entry["interpreted_input"] = reply["interpreted_input"]
    if reply.get("result_kind"):
        entry["result_kind"] = reply["result_kind"]
    return entry


def run(arguments, emit=None, register=None):
    """The completion the engine returns to the client, or an error with its HTTP status. `emit` gets the
    progress events; `register` gets the engine handle, so that a cancellation can stop the turn."""
    request = arguments.get("request") if isinstance(arguments, dict) else None
    if not isinstance(request, dict):
        return {"status": 400, "error": {"message": "request must be a JSON object", "type": "invalid_request_error"}}
    if request.get("tools") or request.get("functions"):
        return {"status": 400, "error": {"message": "requests with tools are answered by the raw endpoint",
                                         "type": "invalid_request_error"}}
    started = time.monotonic()
    if _media(request):
        # the client sends the attachment to the model itself, with its math tools in its own loop
        return {"id": "chatcmpl-tosh-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                "model": str(request.get("model") or "tosh-agent"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "tosh": {"version": 1, "outcome": "declined", "seconds": 0.0, "passes": 0}}
    engine = Engine(str(arguments.get("base_url") or ""), arguments.get("api_key"), arguments.get("agent_key"))
    if register:
        register(engine)
    turn = None
    try:
        turn = Turn(engine, request, emit)
        content, outcome = turn.run()
    except Invalid as error:
        return {"status": 400, "error": {"message": str(error), "type": "invalid_request_error"}}
    except Cancelled:
        return {"status": 499, "error": {"message": "the request was cancelled", "type": "cancelled"}}
    except EngineFailure as error:
        return error.answer(turn.lang if turn else "en")
    except (RuntimeError, OSError, ValueError, http.client.HTTPException) as error:
        return {"status": 502, "error": {"message": str(error)[:300], "type": "engine_error"}}
    finish = "stop"
    # an answer the model wrote until its tokens ran out says it is incomplete
    if outcome == "answered" and turn.finish == "length":
        content, outcome, finish = (content.rstrip() + "\n\n" + CUT.get(turn.lang, CUT["en"])).lstrip(), "cut_at_limit", \
            "length"
    results = policy.ledger(turn.calls)
    return {
        "id": "chatcmpl-tosh-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(request.get("model") or "tosh-agent"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": dict(turn.usage, total_tokens=turn.usage["prompt_tokens"] + turn.usage["completion_tokens"]),
        "tosh": {
            "version": 1,
            "intent": turn.intent,
            "outcome": outcome,
            "calls": [_describe(c) for c in turn.calls],
            "validated_results": [{"tool": r["tool"], "operation": r["operation"],
                                   "result_kind": "exact" if r["exact"] else "approximate",
                                   "result": {k: v for k, v in (policy.reply_of(r["reply"]) or {}).items()
                                              if k not in ("success", "operation", "warnings", "interpreted_input", "result_kind")},
                                   "interpreted_input": r["input"]} for r in results],
            "passes": turn.passes,
            "seconds": round(time.monotonic() - started, 3),
        },
    }

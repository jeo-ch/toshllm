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


class Engine:
    def __init__(self, base_url, api_key, agent_key=""):
        parsed = urllib.parse.urlparse(base_url)
        self.host, self.port = parsed.hostname or "127.0.0.1", parsed.port or 80
        self.api_key = api_key or ""
        # what /tools asks before it lists or runs the tools of a server only the agent may use
        self.agent_key = agent_key or ""
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._connection = None

    def cancel(self):
        """Stops the request in flight: the engine sees the connection close and stops generating."""
        with self._lock:
            self.cancelled.set()
            connection = self._connection
        if connection is not None and connection.sock is not None:
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
            self._connection = connection
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
                self._connection = None
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
        return self._send("POST", "/v1/chat/completions", dict(body, stream=False))

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
            return reply
        if failure:
            return failure[0]
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

    def _body(self, tools, choice, note):
        body = {k: self.request[k] for k in _GENERATION if k in self.request}
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        if body["chat_template_kwargs"].get("enable_thinking") is False:
            body.setdefault("thinking_budget_tokens", 0)
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + self.history
        messages = messages + [{"role": "user", "content": self.question}] + self.messages
        if note:
            messages = messages + [{"role": "user", "content": note}]
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

        def on_delta(field, text):
            nonlocal shown
            shown = shown or field == "content"
            self.emit({"type": "delta", field: text})

        reply = self.engine.stream(body, on_delta) if live else self.engine.complete(body)
        if not isinstance(reply, dict) or not reply.get("choices"):
            error = reply.get("error") if isinstance(reply, dict) and isinstance(reply.get("error"), dict) else {}
            # a request the engine refuses (no model in router mode, a context too small...) is the client's to fix
            if error.get("type") == "invalid_request_error" or error.get("code") == 400:
                raise Invalid(str(error.get("message") or "the engine refused the request"))
            raise RuntimeError(error.get("message") or "the engine gave no answer")
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
        tools = [t["definition"] for t in listed if isinstance(t.get("definition"), dict)]
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
                    {"id": e["id"], "type": "function", "function": {"name": e["name"], "arguments": e["arguments"]}}
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
    engine = Engine(str(arguments.get("base_url") or ""), arguments.get("api_key"), arguments.get("agent_key"))
    if register:
        register(engine)
    try:
        turn = Turn(engine, request, emit)
        content, outcome = turn.run()
    except Invalid as error:
        return {"status": 400, "error": {"message": str(error), "type": "invalid_request_error"}}
    except Cancelled:
        return {"status": 499, "error": {"message": "the request was cancelled", "type": "cancelled"}}
    except (RuntimeError, OSError, ValueError) as error:
        return {"status": 502, "error": {"message": str(error)[:300], "type": "engine_error"}}
    results = policy.ledger(turn.calls)
    return {
        "id": "chatcmpl-tosh-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(request.get("model") or "tosh-agent"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
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

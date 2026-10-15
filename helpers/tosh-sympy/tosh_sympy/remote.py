# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A local MCP server that relays to a remote one, since llama-server only starts servers it can
talk to over stdio. Streamable HTTP first, the older HTTP+SSE transport when the server lacks it.

    python3 -I -B remote.py --url https://host/mcp [--headers-env VAR] [--header-env Name=VAR]
                            [--transport auto|http|sse] [--timeout SECONDS]

Header values come from environment variables the app sets on the engine, never from arguments,
so a token never shows in the process list."""

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

ACCEPT = "application/json, text/event-stream"


def events(stream):
    """Server-sent events as (name, data) pairs."""
    name, data = "message", []
    for raw in stream:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield name, "\n".join(data)
            name, data = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield name, "\n".join(data)


class Relay:
    def __init__(self, url, headers, timeout, transport):
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.transport = transport
        self.session = None
        self.protocol = None
        self.initialize_id = None
        self.endpoint = None
        self.endpoint_ready = threading.Event()
        self.output = threading.Lock()

    def write(self, message):
        if isinstance(message, list):
            for item in message:
                self.write(item)
            return
        if self.initialize_id is not None and message.get("id") == self.initialize_id:
            self.protocol = (message.get("result") or {}).get("protocolVersion")
        with self.output:
            sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    def fail(self, message, reason):
        # an unanswered request would hold the engine until its own timeout
        if isinstance(message, dict) and "id" in message and "method" in message:
            self.write({"jsonrpc": "2.0", "id": message["id"],
                        "error": {"code": -32000, "message": "remote MCP server: " + reason}})

    def request(self, url, method="GET", body=None, accept=ACCEPT):
        headers = dict(self.headers)
        headers["Accept"] = accept
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        if self.protocol:
            headers["MCP-Protocol-Version"] = self.protocol
        return urllib.request.Request(url, data=body, method=method, headers=headers)

    def send(self, message):
        if isinstance(message, dict) and message.get("method") == "initialize":
            self.initialize_id = message.get("id")
        if self.transport == "sse":
            return self.send_sse(message)
        try:
            self.send_http(message)
        except urllib.error.HTTPError as error:
            # a server without Streamable HTTP refuses the first POST; the older transport may still work
            if self.transport == "auto" and self.session is None and error.code in (400, 404, 405):
                self.transport = "sse"
                return self.send_sse(message)
            self.fail(message, "HTTP %d" % error.code)
        except (urllib.error.URLError, OSError, ValueError) as error:
            self.fail(message, str(getattr(error, "reason", error)))

    def send_http(self, message):
        body = json.dumps(message).encode()
        with urllib.request.urlopen(self.request(self.url, "POST", body), timeout=self.timeout) as response:
            if self.transport == "auto":
                self.transport = "http"
            self.session = response.headers.get("Mcp-Session-Id") or self.session
            if response.status == 202:
                return
            if "text/event-stream" in (response.headers.get("Content-Type") or ""):
                for _, data in events(response):
                    if data.strip():
                        self.write(json.loads(data))
                return
            raw = response.read()
            if raw.strip():
                self.write(json.loads(raw))

    def send_sse(self, message):
        if self.endpoint is None:
            threading.Thread(target=self.listen, daemon=True).start()
            if not self.endpoint_ready.wait(self.timeout) or self.endpoint is None:
                return self.fail(message, "no SSE endpoint")
        try:
            body = json.dumps(message).encode()
            with urllib.request.urlopen(self.request(self.endpoint, "POST", body, "application/json"),
                                        timeout=self.timeout) as response:
                response.read()
        except urllib.error.HTTPError as error:
            self.fail(message, "HTTP %d" % error.code)
        except (urllib.error.URLError, OSError, ValueError) as error:
            self.fail(message, str(getattr(error, "reason", error)))

    def listen(self):
        try:
            response = urllib.request.urlopen(self.request(self.url, accept="text/event-stream"), timeout=self.timeout)
            # the stream stays quiet between answers; only connecting may time out
            sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if sock is not None:
                sock.settimeout(None)
            for name, data in events(response):
                if name == "endpoint":
                    self.endpoint = urllib.parse.urljoin(self.url, data.strip())
                    self.endpoint_ready.set()
                elif data.strip():
                    self.write(json.loads(data))
        except (urllib.error.URLError, OSError, ValueError):
            pass
        finally:
            self.endpoint_ready.set()

    def close(self):
        if self.transport == "http" and self.session:
            try:
                urllib.request.urlopen(self.request(self.url, "DELETE"), timeout=5).close()
            except (urllib.error.URLError, OSError):
                pass


def headers_from(arguments, environ):
    headers = {}
    if arguments.headers_env:
        value = environ.get(arguments.headers_env, "")
        loaded = json.loads(value) if value.strip() else {}
        if not isinstance(loaded, dict):
            raise ValueError("%s must hold a JSON object" % arguments.headers_env)
        headers.update({str(k): str(v) for k, v in loaded.items()})
    for pair in arguments.header_env:
        name, _, variable = pair.partition("=")
        if environ.get(variable):
            headers[name] = environ[variable]
    return headers


def main(argv=None, environ=None):
    parser = argparse.ArgumentParser(description="Relay an MCP server over HTTP to stdio.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--headers-env", default="")
    parser.add_argument("--header-env", action="append", default=[])
    parser.add_argument("--transport", choices=["auto", "http", "sse"], default="auto")
    parser.add_argument("--timeout", type=float, default=60.0)
    arguments = parser.parse_args(argv)
    if urllib.parse.urlparse(arguments.url).scheme not in ("http", "https"):
        parser.error("--url must be http or https")
    relay = Relay(arguments.url, headers_from(arguments, os.environ if environ is None else environ),
                  arguments.timeout, arguments.transport)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            relay.send(message)
    finally:
        relay.close()


if __name__ == "__main__":
    main()

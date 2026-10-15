# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the relay to remote MCP servers, against local servers for each transport.

    vendor/tosh-sympy/python/bin/python3 -I helpers/tosh-sympy/tests/test_remote.py
"""

import http.server
import json
import os
import queue
import subprocess
import sys
import threading
import unittest

HELPERS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = os.path.join(HELPERS, "tosh_sympy", "remote.py")
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
READY = {"jsonrpc": "2.0", "method": "notifications/initialized"}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
CALL = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {}}}


def result(message, value):
    return {"jsonrpc": "2.0", "id": message["id"], "result": value}


class StreamableHandler(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((message.get("method"), {k.lower(): v for k, v in self.headers.items()}))
        if "id" not in message:
            self.send_response(202)
            self.end_headers()
            return
        if message["method"] == "initialize":
            self.reply(result(message, {"protocolVersion": "2025-06-18", "capabilities": {}}), {"Mcp-Session-Id": "abc"})
        elif message["method"] == "tools/list":
            body = ("event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"})
                    + "\n\ndata: " + json.dumps(result(message, {"tools": [{"name": "echo"}]})) + "\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.headers.get("Authorization") != "Bearer t":
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.reply(result(message, {"content": [{"type": "text", "text": "ok"}]}))

    def do_DELETE(self):
        type(self).seen.append(("DELETE", {k.lower(): v for k, v in self.headers.items()}))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def reply(self, message, extra=None):
        body = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


class LegacyHandler(http.server.BaseHTTPRequestHandler):
    outbox = queue.Queue()

    def log_message(self, *args):
        pass

    def do_POST(self):
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/sse":
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if "id" in message:
            type(self).outbox.put(result(message, {"echo": message["method"]}))
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"event: endpoint\ndata: /messages?session=1\n\n")
        self.wfile.flush()
        while True:
            message = type(self).outbox.get()
            if message is None:
                return
            self.wfile.write(("data: " + json.dumps(message) + "\n\n").encode())
            self.wfile.flush()


class RemoteTest(unittest.TestCase):
    def serve(self, handler):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return "http://127.0.0.1:%d" % server.server_address[1]

    def relay(self, url, messages, expected):
        env = dict(os.environ, TEST_TOKEN="Bearer t")
        process = subprocess.Popen([sys.executable, "-I", "-B", REMOTE, "--url", url, "--timeout", "5",
                                    "--header-env", "Authorization=TEST_TOKEN"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        for message in messages:
            process.stdin.write((json.dumps(message) + "\n").encode())
        process.stdin.flush()
        lines = [json.loads(process.stdout.readline()) for _ in range(expected)]
        process.stdin.close()
        process.wait(timeout=10)
        process.stdout.close()
        return lines

    def test_streamable_http_keeps_the_session_and_reads_sse_answers(self):
        StreamableHandler.seen = []
        out = self.relay(self.serve(StreamableHandler) + "/mcp", [INIT, READY, LIST, CALL], 4)
        self.assertEqual(out[0]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(out[1]["method"], "notifications/progress")
        self.assertEqual(out[2]["result"]["tools"], [{"name": "echo"}])
        self.assertEqual(out[3]["result"]["content"][0]["text"], "ok")
        call = next(headers for method, headers in StreamableHandler.seen if method == "tools/call")
        self.assertEqual(call.get("mcp-session-id"), "abc")
        self.assertEqual(call.get("mcp-protocol-version"), "2025-06-18")
        self.assertIn("DELETE", [method for method, _ in StreamableHandler.seen])

    def test_a_refused_call_is_answered_instead_of_hanging(self):
        url = self.serve(StreamableHandler) + "/mcp"
        process = subprocess.Popen([sys.executable, "-I", "-B", REMOTE, "--url", url, "--timeout", "5"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, _ = process.communicate((json.dumps(CALL) + "\n").encode(), timeout=10)
        answer = json.loads(out.decode().splitlines()[0])
        self.assertEqual(answer["id"], 3)
        self.assertIn("HTTP 401", answer["error"]["message"])

    def test_the_older_sse_transport_is_used_when_the_server_lacks_streamable_http(self):
        LegacyHandler.outbox = queue.Queue()
        self.addCleanup(lambda: LegacyHandler.outbox.put(None))
        out = self.relay(self.serve(LegacyHandler) + "/sse", [INIT, READY, LIST], 2)
        self.assertEqual([message["id"] for message in out], [1, 2])
        self.assertEqual(out[1]["result"], {"echo": "tools/list"})

    def test_header_values_come_from_the_environment_only(self):
        sys.path.insert(0, HELPERS)
        from tosh_sympy import remote
        arguments = type("A", (), {"headers_env": "H", "header_env": ["X-Key=K", "X-None=MISSING"]})()
        headers = remote.headers_from(arguments, {"H": '{"Authorization": "Bearer a"}', "K": "k"})
        self.assertEqual(headers, {"Authorization": "Bearer a", "X-Key": "k"})


if __name__ == "__main__":
    unittest.main()

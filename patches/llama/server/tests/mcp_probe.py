# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A stdio MCP server for the engine's tests: each tool misbehaves in one way, with no effect outside
this process, and every message it receives is appended, with its pid, to the log named by its first
argument.

    echo   answers its message, after any answer held by defer
    hang   never answers
    defer  holds its answer until the next call is answered
    gate   answers once the file at `path` exists
    slow   stops reading for `seconds`, then answers
    die    exits mid-call
    eof    closes its output and keeps running
"""

import json
import os
import sys
import time

LOG = sys.argv[1]
PID = os.getpid()
TOOLS = [{"name": name, "description": name,
          "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}, "path": {"type": "string"},
                                                            "seconds": {"type": "number"}}}}
         for name in ("echo", "hang", "defer", "gate", "slow", "die", "eof")]


def log(**entry):
    # one write per line, so a reader never sees two lines interleaved
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(dict(entry, pid=PID)) + "\n").encode())
    finally:
        os.close(fd)


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def answer(rid, text):
    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}}


log(started=True)
held = []
for line in sys.stdin:
    message = json.loads(line)
    log(got=message)
    rid = message.get("id")
    if rid is None:
        continue
    method = message.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                                                      "serverInfo": {"name": "probe", "version": "1"}}})
        continue
    if method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        continue
    name = message["params"]["name"]
    arguments = message["params"].get("arguments") or {}
    text = arguments.get("message", "")
    if name == "hang":
        continue
    if name == "defer":
        held.append(answer(rid, "late answer to %s" % text))
        continue
    if name == "die":
        os._exit(3)
    if name == "eof":
        sys.stdout.flush()
        os.close(1)
        continue
    if name == "gate":
        end = time.monotonic() + 60
        while not os.path.exists(arguments["path"]) and time.monotonic() < end:
            time.sleep(0.01)
    if name == "slow":
        time.sleep(arguments.get("seconds", 1))
        log(woke=rid)
    for late in held:
        send(late)
    held.clear()
    send(answer(rid, text))

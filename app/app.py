#!/usr/bin/env python3
"""Mini-Challenge 3 entry point: a thin client for the resident server (server.py).

The harness runs this twice over:

    python3 /app/app.py --index /app/corpus                      once, before any question
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."   once per question

Standard library only, so it starts in milliseconds. A query ALWAYS writes a valid
/app/output/<query-id>_output.json - if anything goes wrong it writes a refusal,
because a missing file or key scores zero while a refusal at least can be right.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
QUERY_WAIT_S = float(os.environ.get("MC3_QUERY_WAIT", 27))     # the harness allows 30 s per question, process start included


def call(req, wait, timeout):
    """One JSON request to the server. Keeps retrying the connection for `wait` seconds: the model may still be loading."""
    end = time.monotonic() + wait
    while True:
        s = socket.socket(socket.AF_UNIX)
        try:
            s.settimeout(timeout)
            s.connect(SOCK)
            break
        except OSError:
            s.close()
            if time.monotonic() > end:
                raise
            time.sleep(0.5)
    with s:
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            d = s.recv(65536)
            if not d:
                break
            buf += d
    return json.loads(buf)


def write_atomic(path, payload):
    # Whole file or nothing: a half-written file read by the harness mid-flush is invalid JSON and scores zero.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        try:
            r = call({"cmd": "index", "corpus": str(args.index)}, wait=540, timeout=590)
        except Exception as e:
            print(f"index failed: server unreachable: {e}", file=sys.stderr)
            return 1
        print(json.dumps(r))
        return 1 if "error" in r else 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    start = time.monotonic()
    out = {"answer": "", "citations": [], "confidence": 0.0}
    try:
        r = call({"cmd": "query", "corpus": str(args.corpus), "query": args.query},
                 wait=QUERY_WAIT_S - 5, timeout=QUERY_WAIT_S)
        if "error" in r:
            raise RuntimeError(r["error"])
        out = {"answer": str(r["answer"]), "citations": [str(c) for c in r["citations"]],
               "confidence": float(r.get("confidence", 0.0))}
    except Exception as e:
        print(f"query failed after {time.monotonic() - start:.1f}s, writing a refusal: {e}", file=sys.stderr)
    write_atomic(OUTPUT_DIR / (args.query_id + "_output.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

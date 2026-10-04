#!/usr/bin/env python3
"""Resident process: holds the model and the index behind a unix socket.

The harness starts a NEW app.py for every question; if that loaded the model or
parsed the corpus, all ten questions would blow the 30 s budget. So app.py is a
thin client and this process does the work once.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import ingest
import llm as backends
import rag

SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
QUERY_BUDGET_S = 24     # the harness allows 30 s per question; app.py gives up shortly after this

model = None
idx = None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def build_index(corpus):
    global idx
    t0 = time.monotonic()
    chunks, files = ingest.ingest(corpus, ocr=backends.reader(model))
    ingest.save(INDEX_DIR, chunks, files)
    idx = rag.Index(chunks)
    skipped = [f for f in files if f["status"] != "ok"]
    log(f"indexed {len(files)} files, {len(chunks)} chunks, skipped {len(skipped)}: "
        + ", ".join(f"{f['path']} ({f['why'] if 'why' in f else f['status']})" for f in skipped))
    return {"files": len(files), "chunks": len(chunks), "skipped": skipped, "seconds": round(time.monotonic() - t0, 1)}


def ask(corpus, question):
    if idx is None:             # --index never ran or failed: index now instead of answering blind
        build_index(corpus)
    t0 = time.monotonic()
    res = rag.answer(idx, model, Path(corpus), question, deadline=t0 + QUERY_BUDGET_S)
    log(f"{time.monotonic() - t0:4.1f}s {question!r} -> {res['answer']!r} {res['citations']}")
    return {k: res[k] for k in ("answer", "citations", "confidence")}


def read_line(conn):
    buf = b""
    while not buf.endswith(b"\n"):
        d = conn.recv(65536)
        if not d:
            break
        buf += d
    return buf


def serve():
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX)
    s.bind(SOCK)
    s.listen(8)
    log(f"ready on {SOCK}")
    while True:
        conn, _ = s.accept()
        with conn:
            try:
                conn.settimeout(10)
                req = json.loads(read_line(conn))
                conn.settimeout(None)
                if req["cmd"] == "index":
                    resp = build_index(req["corpus"])
                elif req["cmd"] == "query":
                    resp = ask(req["corpus"], req["query"])
                else:
                    resp = {"ok": True}
            except Exception as e:      # never die on one bad request
                traceback.print_exc()
                resp = {"error": f"{type(e).__name__}: {e}"}
            try:
                conn.sendall(json.dumps(resp).encode() + b"\n")
            except OSError:             # client already gave up
                pass


def main():
    global model, idx
    t0 = time.monotonic()
    model = backends.make()
    log(f"model loaded in {time.monotonic() - t0:.1f}s")
    try:                    # the first call compiles GPU kernels; do it now, not inside a graded question
        import io

        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), "white").save(buf, "PNG")
        model.generate("Reply with OK.", [buf.getvalue()], 4)
        model.generate("Reply with OK.", [], 4)
        log(f"warmed up after {time.monotonic() - t0:.1f}s")
    except Exception as e:  # not fatal: the first question is just slower
        log(f"warm-up failed: {e}")
    chunks = ingest.load(INDEX_DIR)     # survives a server restart
    if chunks:
        idx = rag.Index(chunks)
        log(f"reloaded index: {len(chunks)} chunks")
    serve()


if __name__ == "__main__":
    main()

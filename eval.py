#!/usr/bin/env python3
"""Score the pipeline the way the grader does: normalised answer AND exact citation set, 20 points each.

    python3 eval.py --llm ollama                        # in-process, local stand-in model (text only)
    python3 eval.py --llm qwen                          # in-process, the real model (needs the GPU)
    python3 eval.py --cli                               # end to end through app.py and a running server.py
    python3 eval.py --corpus tests/extra --questions tests/extra/questions.json ...

--ocr-fixtures FILE maps image paths to hand-written transcriptions: a stand-in for the vision model when
the backend cannot see pictures. Dev only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "app"))
import ingest  # noqa: E402
import rag  # noqa: E402


def norm(s):
    """The grader's normalisation: upper case, whitespace and - . · _ removed."""
    return re.sub(r"[\s\-._·]", "", s.upper())


def score(q, answer, citations):
    ok_answer = norm(answer) in {norm(a) for a in [q["expected_answer"], *q.get("answer_aliases", [])]}
    if q["expected_answer"] == "":
        ok_answer = answer == ""
    ok_cites = set(citations) == set(q["expected_citations"])
    return ok_answer, ok_cites


def fixture_ocr(corpus, fixtures):
    by_hash = {}
    for rel, text in json.loads(Path(fixtures).read_text()).items():
        by_hash[hashlib.sha1((Path(corpus) / rel).read_bytes()).hexdigest()] = text
    return lambda data: by_hash.get(hashlib.sha1(data).hexdigest(), "")


def run_cli(corpus, question, qid):
    out = Path(tempfile.mkdtemp())
    env = {**os.environ, "MC2_OUTPUT_DIR": str(out)}
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, str(HERE / "app" / "app.py"), "--corpus", str(corpus),
                        "--query-id", qid, "--query", question], env=env, capture_output=True, text=True)
    if p.stderr.strip():
        print("   stderr:", p.stderr.strip()[:300])
    r = json.loads((out / f"{qid}_output.json").read_text())
    return r, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "mc3-corpus"))
    ap.add_argument("--questions", default=str(HERE / "sample-questions.json"))
    ap.add_argument("--llm", default="ollama", choices=["ollama", "qwen"])
    ap.add_argument("--cli", action="store_true", help="go through app.py and a running server instead of in-process")
    ap.add_argument("--ocr-fixtures")
    ap.add_argument("--only", type=int, nargs="*", help="question numbers to run")
    ap.add_argument("--trace", action="store_true", help="print the model's raw replies")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    qs = json.loads(Path(args.questions).read_text())["queries"]
    if args.only:
        qs = [q for q in qs if q["n"] in args.only]

    if not args.cli:
        import llm as backends
        model = backends.make(args.llm)
        ocr = fixture_ocr(corpus, args.ocr_fixtures) if args.ocr_fixtures else backends.reader(model)
        t0 = time.monotonic()
        chunks, files = ingest.ingest(corpus, ocr=ocr)
        idx = rag.Index(chunks)
        print(f"indexed {len(files)} files / {len(chunks)} chunks in {time.monotonic() - t0:.1f}s; "
              f"skipped: {[f['path'] for f in files if f['status'] != 'ok']}\n")

    total = 0
    for q in qs:
        if args.cli:
            r, secs = run_cli(corpus, q["query"], f"query_{q['n']:02d}")
        else:
            t0 = time.monotonic()
            r = rag.answer(idx, model, corpus, q["query"], deadline=time.monotonic() + 24)
            secs = time.monotonic() - t0
        a_ok, c_ok = score(q, r["answer"], r["citations"])
        pts = 20 if a_ok and c_ok else 0
        total += pts
        verdict = "PASS" if pts else "FAIL" + ("" if a_ok else " answer") + ("" if c_ok else " citations")
        print(f"Q{q['n']:<2} {verdict:<22} {secs:5.1f}s  answer={r['answer']!r} cites={r['citations']}")
        if not pts:
            print(f"    wanted answer={q['expected_answer']!r} cites={q['expected_citations']}")
        if args.trace and "trace" in r:
            for i, rd in enumerate(r["trace"]["rounds"], 1):
                print(f"    --- round {i} evidence: {rd['evidence']}\n{rd['raw']}")
            if "why" in r["trace"]:
                print("    why refused:", r["trace"]["why"])
    print(f"\nscore {total}/{20 * len(qs)}")


if __name__ == "__main__":
    main()

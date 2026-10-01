"""Offline tests: no GPU, no real model. A scripted fake model stands in, so what is tested is everything
around it - hostile files, retrieval, citation selection, refusals, and the socket plumbing.

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "app"))
import ingest  # noqa: E402
import rag  # noqa: E402

CORPUS = HERE / "mc3-corpus"
QUERIES = {q["n"]: q for q in json.loads((HERE / "sample-questions.json").read_text())["queries"]}


def evidence(prompt):
    """[(number, file, text)] parsed back out of a rendered prompt."""
    body = prompt.split("Evidence:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    out = []
    for m in re.finditer(r"^\[(\d+)\] (\S+) \(.*?\)[^\n]*\n(.*?)(?=\n\n\[\d+\] |\Z)", body, re.S | re.M):
        out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


class Scripted:
    """Fake model: each reply is a function of the evidence it was shown, or a plain string."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, images=(), max_new_tokens=64):
        if "Evidence:\n" not in prompt:         # an OCR request at index time: the fake model reads nothing
            return ""
        self.prompts.append((prompt, list(images)))
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return r(evidence(prompt)) if callable(r) else r


def cites(ev, needle):
    return ",".join(str(n) for n, _, t in ev if needle in t) or "NONE"


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.files = ingest.ingest(CORPUS)
        cls.idx = rag.Index(cls.chunks)

    def ask(self, n, llm):
        return rag.answer(self.idx, llm, CORPUS, QUERIES[n]["query"])


class Units(unittest.TestCase):
    def test_terms_keep_identifiers_findable_in_every_spelling(self):
        t = set(rag.terms("TQ-40 THERM_ALERT# fixed 4.3.2."))
        for want in ("tq-40", "tq", "40", "tq40", "therm_alert", "thermalert", "fix", "4.3.2", "432"):
            self.assertIn(want, t)
        self.assertTrue(set(rag.terms("TQ-40")) & set(rag.terms("TQ40")))
        self.assertTrue(set(rag.terms("field-replaceable")) & set(rag.terms("field replaceable")))

    def test_identifiers(self):
        got = rag.ids("incident logged against ORR-1847 on inference-node-07, code E7731, firmware 4.3.2, DEFAULT_BATCH_TIMEOUT_S")
        self.assertEqual(set(got), {"ORR-1847", "inference-node-07", "E7731"})

    def test_chunks_stay_under_what_the_model_is_shown(self):
        for lines in (["x" * 10000], ["y" * 900] * 7, ["short line"] * 500, ["a" * 3000, "b" * 3000, "c" * 50]):
            for _, _, text in ingest.windows(lines):
                self.assertLessEqual(len(text), rag.SHOW)
        text = "".join(t for _, _, t in ingest.windows(["z" * 10000]))
        self.assertGreaterEqual(len(text), 10000)           # nothing dropped

    def test_flat_matches_across_formatting(self):
        self.assertEqual(rag.flat("Q3 FY27"), rag.flat("q3-fy27"))

    def test_whole_token_match(self):
        self.assertFalse(rag.has("board power 350 W", "50"))
        self.assertFalse(rag.has("rev 1940", "94"))
        self.assertTrue(rag.has("lead_time_days: 50", "50"))
        self.assertTrue(rag.has("fixed_in: 4.3.2", "4.3.2"))
        self.assertTrue(rag.has("enters sampling in Q3FY27.", "Q3 FY27"))
        self.assertTrue(rag.has("part_number: ORR-FAN-2214-B | x", "ORR-FAN-2214-B"))
        self.assertFalse(rag.has("nothing here", ""))

    def test_units_are_dropped_from_numbers_only(self):
        for given, want in [("94 C", "94"), ("88°C", "88"), ("225 W", "225"), ("225W", "225"), ("180 seconds", "180"),
                            ("56 GiB", "56"), ("10,000 USD", "10,000"), ("3.5 GHz", "3.5"), ("94", "94")]:
            self.assertEqual(rag.tidy(given), want, given)
        for keep in ("Q3 FY27", "4.3.2", "B14", "REV-C2", "ORR-FAN-2214-B", "12V-2x6", "2A", "12V", "E7731", "FY27"):
            self.assertEqual(rag.tidy(keep), keep, keep)


class HostileCorpus(Base):
    def test_sample_corpus(self):
        status = {f["path"]: f["status"] for f in self.files}
        self.assertEqual(status["vendor/internal_audit.txt"], "skipped")             # chmod 000
        self.assertEqual(status["vendor/supplier_agreement_ENCRYPTED.pdf"], "skipped")
        self.assertEqual(status["vendor/telemetry_capture.dat"], "skipped")          # unknown type
        self.assertEqual(status["specs/tq40_datasheet_r1_WITHDRAWN.pdf"], "withdrawn")
        self.assertEqual(status["specs/tq40_datasheet_r2.pdf"], "ok")                # says "Revision 1 is withdrawn" but is current
        for f in ("support/bug_database.csv", "support/rma_parts.xlsx", "planning/roadmap_fy27.docx",
                  "logs/prod_inference_2026-09-02.log", "engineering/ingest_service.py",
                  "specs/backplane_pinout.png", "support/asset_label.jpg"):
            self.assertEqual(status[f], "ok", f)

    def test_garbage_never_stops_the_walk(self):
        from pypdf import PdfReader, PdfWriter
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a_empty_dir").mkdir()
            (d / "a_corrupt.pdf").write_bytes(os.urandom(500))
            (d / "b_corrupt.docx").write_bytes(os.urandom(500))
            (d / "c_corrupt.xlsx").write_bytes(os.urandom(500))
            (d / "d_zero.csv").write_bytes(b"")
            (d / "e_binary.txt").write_bytes(b"\0\1\2\3" * 100)
            (d / "f_corrupt.png").write_bytes(b"not a png")
            w = PdfWriter()                                                   # an encrypted PDF that opens but cannot be read
            w.add_blank_page(200, 200)
            w.encrypt("secret", algorithm="RC4-128")
            with open(d / "g_encrypted.pdf", "wb") as f:
                w.write(f)
            (d / "h_locked_dir").mkdir()
            (d / "h_locked_dir" / "x.txt").write_text("unreachable")
            os.chmod(d / "h_locked_dir", 0)
            (d / "z_good.txt").write_text("the only readable file, ORR-9 is here")
            try:
                chunks, files = ingest.ingest(d)
            finally:
                os.chmod(d / "h_locked_dir", 0o755)
            self.assertEqual({c["file"] for c in chunks} - {"f_corrupt.png"}, {"z_good.txt"})
            self.assertIn("z_good.txt", {f["path"] for f in files if f["status"] == "ok"})


class Retrieval(Base):
    def top(self, n):
        return [self.chunks[i]["file"] for i in self.idx.search(QUERIES[n]["query"], rag.K)]

    def test_expected_file_ranks_first(self):
        for n in (1, 2, 3, 4, 5, 6, 8):
            self.assertEqual(self.top(n)[0], QUERIES[n]["expected_citations"][0], f"Q{n}")

    def test_withdrawn_never_retrieved(self):
        for n in QUERIES:
            self.assertNotIn("specs/tq40_datasheet_r1_WITHDRAWN.pdf", self.top(n))

    def test_identifier_link_reaches_the_ticket_row(self):
        ev = self.idx.search(QUERIES[9]["query"], rag.K)
        got = {(self.chunks[b]["file"], self.chunks[b]["loc"]) for _, b, x in self.idx.links(ev, QUERIES[9]["query"]) if x == "ORR-1847"}
        self.assertIn(("support/bug_database.csv", "r2"), got)

    def test_most_specific_identifier_first(self):
        ev = self.idx.search(QUERIES[9]["query"], rag.K)
        self.assertEqual(self.idx.links(ev, QUERIES[9]["query"])[0][2], "ORR-1847")


class Pipeline(Base):
    def test_single_file(self):
        r = self.ask(1, Scripted(lambda ev: f"ANSWER: 94\nSOURCES: {cites(ev, '94 C')}"))
        self.assertEqual((r["answer"], r["citations"]), ("94", ["specs/tq40_datasheet_r2.pdf"]))

    def test_two_file_chain_cites_both(self):
        r = self.ask(9, Scripted(lambda ev: f"ANSWER: 4.3.2\nSOURCES: {cites(ev, 'ORR-1847')}"))
        self.assertEqual(r["answer"], "4.3.2")
        self.assertEqual(set(r["citations"]), set(QUERIES[9]["expected_citations"]))

    def test_chain_via_need_round(self):
        llm = Scripted(lambda ev: f"ANSWER: NONE\nSOURCES: {cites(ev, 'incident logged')}\nNEED: ORR-1847",
                       lambda ev: f"ANSWER: 4.3.2\nSOURCES: {cites(ev, 'fixed_in: 4.3.2')}")
        r = self.ask(9, llm)
        self.assertEqual(set(r["citations"]), set(QUERIES[9]["expected_citations"]))

    def test_chain_cites_the_log_even_if_the_model_forgets_it(self):
        r = self.ask(9, Scripted(lambda ev: f"ANSWER: 4.3.2\nSOURCES: {cites(ev, 'fixed_in: 4.3.2')}"))
        self.assertEqual(set(r["citations"]), set(QUERIES[9]["expected_citations"]))

    def test_identifier_in_the_question_is_not_a_second_source(self):
        r = self.ask(4, Scripted(lambda ev: f"ANSWER: 4.3.2\nSOURCES: {cites(ev, 'ORR-1847')}"))   # model also cites the log
        self.assertEqual(r["citations"], ["support/bug_database.csv"])

    def test_hallucinated_answer_is_refused(self):                  # Q10: the price exists only in the encrypted file
        r = self.ask(10, Scripted("ANSWER: 6412\nSOURCES: 1\nNEED: NONE"))
        self.assertEqual((r["answer"], r["citations"]), ("", []))

    def test_none_is_refused(self):
        r = self.ask(10, Scripted("ANSWER: NONE\nSOURCES: NONE\nNEED: NONE"))
        self.assertEqual((r["answer"], r["citations"]), ("", []))

    def test_unparseable_reply_is_refused(self):
        r = self.ask(1, Scripted("I think the answer is probably about 94 degrees."))
        self.assertEqual((r["answer"], r["citations"]), ("", []))

    def test_answer_read_off_a_picture_is_accepted(self):
        def reply(ev):
            n = next(n for n, f, _ in ev if f == "specs/backplane_pinout.png")
            return f"ANSWER: B14\nSOURCES: {n}"
        llm = Scripted(reply)
        r = self.ask(7, llm)       # the picture has no text in this index (no OCR here); the model saw it attached
        self.assertEqual((r["answer"], r["citations"]), ("B14", ["specs/backplane_pinout.png"]))
        self.assertTrue(llm.prompts[0][1] and llm.prompts[0][1][0].endswith("backplane_pinout.png"))

    def test_withdrawn_document_is_never_shown_to_the_model(self):
        llm = Scripted("ANSWER: NONE\nSOURCES: NONE\nNEED: NONE")
        self.ask(1, llm)
        self.assertNotIn("WITHDRAWN", llm.prompts[0][0])

    def test_markdown_decorated_reply_parses(self):
        self.assertEqual(rag.parse("**ANSWER:** `94`\n**SOURCES:** 1, 3\n**NEED:** None"), ("94", [1, 3], ""))
        self.assertEqual(rag.parse("<think>hmm</think>\nANSWER: NONE\nSOURCES: NONE\nNEED: ORR-1847"), ("", [], "ORR-1847"))

    def test_deadline_stops_the_loop(self):
        llm = Scripted("ANSWER: NONE\nSOURCES: NONE\nNEED: ORR-1847")
        r = rag.answer(self.idx, llm, CORPUS, QUERIES[9]["query"], deadline=time.monotonic() - 1)
        self.assertEqual((r["answer"], llm.prompts), ("", []))


class Plumbing(unittest.TestCase):
    """app.py -> unix socket -> server.py, with the fake model."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = {**os.environ, "MC3_SOCK": str(cls.tmp / "s.sock"), "MC2_OUTPUT_DIR": str(cls.tmp / "out"),
                   "MC3_QUERY_WAIT": "4"}
        os.environ["MC3_SOCK"] = cls.env["MC3_SOCK"]
        os.environ["MC3_INDEX_DIR"] = str(cls.tmp / "index")
        import server
        cls.server = server
        server.model = Scripted(lambda ev: f"ANSWER: 94\nSOURCES: {cites(ev, '94 C')}")
        threading.Thread(target=server.serve, daemon=True).start()
        for _ in range(50):
            if Path(cls.env["MC3_SOCK"]).exists():
                break
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def app(self, *args, env=None):
        return subprocess.run([sys.executable, str(HERE / "app" / "app.py"), *args], capture_output=True, text=True,
                              env=env or self.env, timeout=60)

    def out(self, qid):
        return json.loads((self.tmp / "out" / f"{qid}_output.json").read_text())

    def test_index_then_query(self):
        p = self.app("--index", str(CORPUS))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["chunks"], len(ingest.ingest(CORPUS)[0]))
        p = self.app("--corpus", str(CORPUS), "--query-id", "query_07", "--query", QUERIES[1]["query"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.out("query_07"), {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9})

    def test_server_down_still_writes_a_valid_refusal(self):
        env = {**self.env, "MC3_SOCK": str(self.tmp / "nobody.sock")}
        t0 = time.monotonic()
        p = self.app("--corpus", str(CORPUS), "--query-id", "query_99", "--query", "anything", env=env)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(self.out("query_99"), {"answer": "", "citations": [], "confidence": 0.0})
        self.assertLess(time.monotonic() - t0, 10)

    def test_server_error_still_writes_a_valid_refusal(self):
        self.server.model = None                  # the next query raises inside the server
        try:
            p = self.app("--corpus", str(CORPUS), "--query-id", "query_98", "--query", "x")
        finally:
            self.server.model = Scripted(lambda ev: f"ANSWER: 94\nSOURCES: {cites(ev, '94 C')}")
        self.assertEqual(p.returncode, 0)
        self.assertEqual(self.out("query_98")["answer"], "")


if __name__ == "__main__":
    unittest.main()

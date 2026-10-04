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
from unittest import mock

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
        got = {(self.chunks[b]["file"], self.chunks[b]["loc"]) for _, b, x in self.idx.links(ev, QUERIES[9]["query"]) if x == "orr-1847"}
        self.assertIn(("support/bug_database.csv", "r2"), got)

    def test_most_specific_identifier_first(self):
        ev = self.idx.search(QUERIES[9]["query"], rag.K)
        self.assertEqual(self.idx.links(ev, QUERIES[9]["query"])[0][2], "orr-1847")


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
        with mock.patch.object(rag, "K", 3), mock.patch.object(rag, "EXTRA", 0):   # a first retrieval that does NOT already hold the ticket row
            r = self.ask(9, llm)
        self.assertEqual(len(llm.prompts), 2)                                       # it really took a second round
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


class ThreeStepChain(unittest.TestCase):
    """log -> ticket row -> release notes: every file on the way is necessary."""

    def setUp(self):
        mk = lambda f, text: {"file": f, "loc": "l1", "kind": "text", "text": text, "status": "ok"}   # noqa: E731
        self.chunks = [
            mk("logs/node-3.log", "2026-08-14 node-3 ERROR E5127 thermal throttle engaged. incident logged against HLX-1042"),
            mk("support/bugs.csv", "ticket: HLX-1042 | component: power | summary: sensor reads high | fixed_in: 5.1.2"),
            mk("support/bugs.csv", "ticket: HLX-1050 | component: pcie | summary: doorbell hang | fixed_in: 5.1.2"),
            mk("support/bugs.csv", "ticket: HLX-1061 | component: thermal | summary: fan curve | fixed_in: 5.1.2"),
            mk("engineering/release_notes_5.1.2.txt", "Halcyon firmware 5.1.2 - release notes\nReleased: 2026-03-12"),
            mk("engineering/release_notes_5.1.1.txt", "Halcyon firmware 5.1.1 - release notes\nReleased: 2026-02-11"),
        ]
        # filler that matches the question's words better than the release notes do, so the first search misses them
        self.chunks += [mk(f"misc/memo{i}.txt", "incident firmware defect fixed published date node log underlying") for i in range(10)]
        self.idx = rag.Index(self.chunks)
        self.q = "The node-3 log from 2026-08-14 shows an incident. On what date was the firmware that fixed the defect published?"

    def test_all_three_files_are_cited(self):
        llm = Scripted(lambda ev: f"ANSWER: NONE\nSOURCES: {cites(ev, 'HLX-1042')}\nNEED: 5.1.2",
                       lambda ev: f"ANSWER: 2026-03-12\nSOURCES: {cites(ev, 'Released: 2026-03-12')}")
        r = rag.answer(self.idx, llm, CORPUS, self.q)
        self.assertEqual(r["answer"], "2026-03-12")
        self.assertEqual(set(r["citations"]), {"logs/node-3.log", "support/bugs.csv", "engineering/release_notes_5.1.2.txt"})

    def test_a_decoy_the_model_also_cites_is_not_a_step(self):
        llm = Scripted(lambda ev: f"ANSWER: NONE\nSOURCES: {cites(ev, 'HLX-1042')}\nNEED: 5.1.2",
                       lambda ev: "ANSWER: 2026-03-12\nSOURCES: " + ",".join(str(n) for n, _, t in ev if "Released" in t))   # cites both notes files
        r = rag.answer(self.idx, llm, CORPUS, self.q)
        self.assertNotIn("engineering/release_notes_5.1.1.txt", r["citations"])


def mk(f, text, kind="text"):
    return {"file": f, "loc": "l1", "kind": kind, "text": text, "status": "ok"}


class RankGateAndLinks(unittest.TestCase):
    def test_file_names_are_search_terms(self):
        self.assertEqual(rag.path_words("engineering/release_notes_5.3.0.txt"), "engineering release notes 5.3.0")
        self.assertIn("5.3.0", rag.terms(rag.path_words("engineering/release_notes_5.3.0.txt")))
        self.assertIn("node-2", rag.terms(rag.path_words("logs/node-2_2026-08-11.log")))
        self.assertIn("2026-08-11", rag.terms(rag.path_words("logs/node-2_2026-08-11.log")))

    def test_a_shared_part_number_does_not_make_an_over_cited_file_a_step(self):
        chunks = [mk("specs/pl-4_datasheet_r2.pdf", 'Halcyon PL-4 "Merlin" datasheet. Board power (TBP) 225 W. Fan assembly part HLX-FAN-1111-B'),
                  mk("support/rma_parts.xlsx", "part_number: HLX-FAN-1111-B | description: Fan assembly | compatible_with: PL-4 | lead_time_days: 42")]
        chunks += [mk(f"misc/note{i}.txt", f"unrelated note number {i} about parking and catering") for i in range(6)]
        llm = Scripted(lambda ev: f"ANSWER: 225 W\nSOURCES: {cites(ev, '225 W')},{cites(ev, 'lead_time_days')}")      # the model over-cites
        r = rag.answer(rag.Index(chunks), llm, CORPUS, "What is the board power of the Merlin?")
        self.assertEqual((r["answer"], r["citations"]), ("225", ["specs/pl-4_datasheet_r2.pdf"]))

    def test_version_in_a_ticket_row_leads_to_the_release_notes_file_named_after_it(self):
        versions = ["5.1.0", "5.1.1", "5.1.2", "5.2.0", "5.2.1", "5.2.2", "5.3.0"]
        chunks = [mk("logs/node-2_2026-08-11.log", "2026-08-11T03:30:00Z node-2 ERROR E5127 thermal throttle. incident logged against HLX-1042"),
                  mk("support/bugs_2025.csv", "ticket: HLX-1042 | component: power | summary: sensor reads high | fixed_in: 5.3.0"),
                  mk("support/bugs_2025.csv", "ticket: HLX-1043 | component: pcie | summary: doorbell hang | fixed_in: 5.2.2")]
        chunks += [mk(f"engineering/release_notes_{v}.txt", f"Halcyon firmware {v} - release notes\nReleased: 2026-{i + 1:02d}-{10 + i:02d}") for i, v in enumerate(versions)]
        q = "The production log for node-2 on 2026-08-11 shows an incident. On what date was the firmware release that fixed the underlying defect published?"
        with mock.patch.object(rag, "K", 2):            # a first retrieval that cannot hold the right release notes
            idx = rag.Index(chunks)
            keys = {(chunks[a]["file"], chunks[b]["file"], x) for a, b, x in idx.links(idx.search(q, 2), q)}
            self.assertIn(("logs/node-2_2026-08-11.log", "support/bugs_2025.csv", "hlx-1042"), keys)
            self.assertIn(("support/bugs_2025.csv", "engineering/release_notes_5.3.0.txt", "5.3.0"), keys)
            llm = Scripted(lambda ev: f"ANSWER: 2026-07-16\nSOURCES: {cites(ev, 'HLX-1042')},{cites(ev, 'Released: 2026-07-16')}")
            r = rag.answer(idx, llm, CORPUS, q)
        self.assertEqual(r["answer"], "2026-07-16")
        self.assertEqual(set(r["citations"]), {"logs/node-2_2026-08-11.log", "support/bugs_2025.csv", "engineering/release_notes_5.3.0.txt"})


class AnchoredAnswers(unittest.TestCase):
    """The model's SOURCES may list only the earlier steps of a chain and forget the item that holds the value."""

    def setUp(self):
        versions = ["5.1.0", "5.1.1", "5.1.2", "5.2.0", "5.2.1", "5.2.2", "5.3.0"]
        self.chunks = [mk("logs/node-2_2026-08-11.log", "2026-08-11T03:30:00Z node-2 ERROR E5127 thermal throttle. incident logged against HLX-1042"),
                       mk("support/bugs_2025.csv", "ticket: HLX-1042 | component: power | summary: sensor reads high | fixed_in: 5.3.0")]
        self.chunks += [mk(f"engineering/release_notes_{v}.txt", f"Halcyon firmware {v} - release notes\nReleased: 2026-{i + 1:02d}-{10 + i:02d}") for i, v in enumerate(versions)]
        self.q = "Node-2 logged an incident on 2026-08-11. When was the firmware version that corrected the root cause made available?"

    def test_the_value_behind_a_cited_pointer_is_accepted_and_all_three_files_cited(self):
        with mock.patch.object(rag, "K", 2):
            idx = rag.Index(self.chunks)
            llm = Scripted(lambda ev: f"ANSWER: 2026-07-16\nSOURCES: {cites(ev, 'HLX-1042')}")     # names the log and the ticket row, not the notes
            r = rag.answer(idx, llm, CORPUS, self.q)
        self.assertEqual(r["answer"], "2026-07-16")
        self.assertEqual(set(r["citations"]), {"logs/node-2_2026-08-11.log", "support/bugs_2025.csv", "engineering/release_notes_5.3.0.txt"})
        self.assertEqual(len(llm.prompts), 1)



class ChoosingTheSource(unittest.TestCase):
    """Two chunks hold the value; the right one is the record the question points at, not a topical neighbour."""

    def setUp(self):
        self.notes = [mk(f"engineering/release_notes_{v}.txt", f"Halcyon firmware {v} - release notes\nScheduler tuning.") for v in ("5.1.0", "5.2.0", "5.2.1")]
        self.log = mk("logs/node-12_2026-08-26.log", "2026-08-26T03:30:00Z node-12 ERROR E5127 thermal throttle. incident logged against HLX-2001")
        self.row = mk("support/bugs_2026.csv", "ticket: HLX-2001 | component: power | summary: sensor reads high | fixed_in: 5.2.0")
        self.chunks = [self.log, self.row] + self.notes + [mk(f"misc/n{i}.txt", f"unrelated {i}") for i in range(6)]

    def both_cited(self, ev):
        return "ANSWER: 5.2.0\nSOURCES: " + ",".join(str(n) for n, _, t in ev if "fixed_in: 5.2.0" in t or "5.2.0 - release notes" in t)

    def test_the_ticket_row_beats_release_notes_that_merely_carry_the_version_in_their_title(self):
        q = "The production log for node-12 on 2026-08-26 shows an incident. Which firmware release fixed the underlying defect?"
        with mock.patch.object(rag, "K", 3):            # as in a real corpus: the ticket row is NOT in the first retrieval, only reachable from the log
            r = rag.answer(rag.Index(self.chunks), Scripted(self.both_cited), CORPUS, q)
        self.assertEqual(r["answer"], "5.2.0")
        self.assertEqual(set(r["citations"]), {"support/bugs_2026.csv", "logs/node-12_2026-08-26.log"})

    def test_a_ticket_id_in_the_question_makes_its_row_the_source(self):
        q = "Which firmware version fixed ticket HLX-2001?"
        r = rag.answer(rag.Index(self.chunks), Scripted(self.both_cited), CORPUS, q)
        self.assertEqual((r["answer"], r["citations"]), ("5.2.0", ["support/bugs_2026.csv"]))


class TwoLogsTwoTickets(unittest.TestCase):
    def test_the_chain_starts_from_the_log_the_question_names(self):
        # both tickets happen to be fixed in 5.2.0, so BOTH rows hold the answer; only one chain starts at the log asked about
        chunks = [mk("logs/node-4_2026-08-26.log", "2026-08-26T03:30:00Z node-4 ERROR E7044 link retrain failed. incident logged against HLX-1406"),
                  mk("logs/node-12_2026-08-26.log", "2026-08-26T03:30:00Z node-12 ERROR E5127 thermal throttle. incident logged against HLX-1290"),
                  mk("support/bugs_2026.csv", "ticket: HLX-1290 | component: power | summary: sensor reads high | fixed_in: 5.2.0"),
                  mk("support/bugs_2026.csv", "ticket: HLX-1406 | component: pcie | summary: doorbell hang | fixed_in: 5.2.0")]
        chunks += [mk(f"misc/n{i}.txt", f"unrelated {i}") for i in range(6)]
        q = "Node-12 had an incident on 2026-08-26 according to its production log. In which firmware version was the root cause corrected?"
        with mock.patch.object(rag, "K", 2):
            llm = Scripted(lambda ev: f"ANSWER: 5.2.0\nSOURCES: {cites(ev, 'fixed_in: 5.2.0')}")        # cites both rows
            r = rag.answer(rag.Index(chunks), llm, CORPUS, q)
        self.assertEqual(r["answer"], "5.2.0")
        self.assertEqual(set(r["citations"]), {"support/bugs_2026.csv", "logs/node-12_2026-08-26.log"})


class SelfCorrection(Base):
    def test_an_answer_from_an_uncited_item_gets_one_chance_to_be_fixed(self):
        llm = Scripted(lambda ev: f"ANSWER: 94\nSOURCES: {cites(ev, 'Q3 FY27')}",          # 94 is in the datasheet, but it cited the roadmap
                       lambda ev: f"ANSWER: 94\nSOURCES: {cites(ev, '94 C')}")
        r = self.ask(1, llm)
        self.assertEqual((r["answer"], r["citations"]), ("94", ["specs/tq40_datasheet_r2.pdf"]))
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("does not appear in the evidence items you cited", llm.prompts[1][0])

    def test_a_model_that_keeps_misciting_is_still_right_when_the_value_is_in_the_evidence(self):
        llm = Scripted(lambda ev: f"ANSWER: 94\nSOURCES: {cites(ev, 'Q3 FY27')}")           # never fixes its numbering
        r = self.ask(1, llm)
        self.assertEqual((r["answer"], r["citations"], len(llm.prompts)), ("94", ["specs/tq40_datasheet_r2.pdf"], 2))

    def test_an_answer_that_is_nowhere_in_the_evidence_needs_no_correction_round(self):
        llm = Scripted("ANSWER: 6412\nSOURCES: 1")
        r = self.ask(10, llm)
        self.assertEqual((r["answer"], r["citations"], len(llm.prompts)), ("", [], 1))


class PromptSwitches(unittest.TestCase):
    def setUp(self):
        self.chunks = [mk("specs/a.pdf", "a"), mk("img/linked.png", "label text", "image"), mk("img/found.png", "pin text", "image")]
        self.idx = rag.Index(self.chunks)

    def test_a_picture_only_a_link_pulled_in_is_not_attached_when_linked_images_is_off(self):
        with mock.patch.object(rag, "LINKED_IMAGES", False):
            _, images = rag.render(self.idx, CORPUS, "q", [0, 1, 2], first={0, 2})        # picture 1 came from a link, picture 2 from the question
        self.assertEqual([Path(i).name for i in images], ["found.png"])
        with mock.patch.object(rag, "LINKED_IMAGES", True):
            _, images = rag.render(self.idx, CORPUS, "q", [0, 1, 2], first={0, 2})
        self.assertEqual([Path(i).name for i in images], ["linked.png", "found.png"])

    def test_the_property_rule_can_be_switched(self):
        for mode, needle in (("long", "never take it from a record about a different one"), ("short", "not with the identifier that led you there"), ("off", None)):
            with mock.patch.object(rag, "PROPERTY_RULE", mode):
                prompt, _ = rag.render(self.idx, CORPUS, "q", [0])
            self.assertEqual(needle in prompt if needle else "specific property" in prompt or "property the question asks for" in prompt, bool(needle), mode)


class RefusalWithAPicture(unittest.TestCase):
    def setUp(self):
        self.chunks = [mk("specs/tq40_datasheet.pdf", "TQ-40 datasheet. Maximum junction temperature .......... 94 C"),
                       mk("support/label_tq40.png", "", "image")]                  # the question finds the picture by its name
        self.q = "What is the maximum junction temperature of the TQ-40, as on its label?"

    def test_a_refusal_with_a_picture_attached_is_asked_again_without_it(self):
        llm = Scripted("ANSWER: NONE\nSOURCES: NONE\nNEED: NONE", lambda ev: f"ANSWER: 94\nSOURCES: {cites(ev, '94 C')}")
        r = rag.answer(rag.Index(self.chunks), llm, CORPUS, self.q)
        self.assertEqual((r["answer"], r["citations"]), ("94", ["specs/tq40_datasheet.pdf"]))
        self.assertEqual(len(llm.prompts), 2)
        self.assertTrue(llm.prompts[0][1])                  # the first attempt had the picture in front of it
        self.assertFalse(llm.prompts[1][1])                 # the second did not

    def test_a_refusal_without_a_picture_is_not_repeated(self):
        llm = Scripted("ANSWER: NONE\nSOURCES: NONE\nNEED: NONE")
        r = rag.answer(rag.Index(self.chunks[:1]), llm, CORPUS, self.q)             # no picture in the evidence
        self.assertEqual((r["answer"], r["citations"], len(llm.prompts)), ("", [], 1))


class StubBackend(Base):
    def test_the_stub_refuses_everything_and_never_needs_torch(self):
        import llm
        model = llm.make("stub")
        r = rag.answer(self.idx, model, CORPUS, QUERIES[1]["query"])
        self.assertEqual((r["answer"], r["citations"]), ("", []))
        self.assertNotIn("torch", sys.modules)             # the whole offline suite runs without importing torch


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

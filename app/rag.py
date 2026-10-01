"""BM25 retrieval, identifier links between files, and the answer loop.

The model is injected (anything with .generate(prompt, images, max_new_tokens)),
so nothing here imports torch and the whole pipeline runs on a laptop.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

K = 8           # chunks shown to the model after the first search
PER_FILE = 3    # at most this many of them from one file
EXTRA = 4       # chunks added by following identifiers into other files
HOPS = 3        # model rounds; a round may ask for one more identifier to be looked up
SHOW = 1600     # characters of one chunk the model sees (ingest keeps chunks under this)

REFUSE = {"answer": "", "citations": [], "confidence": 0.0}

STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did what
which who whom whose when where why how that this these those it its as into than then there their they them
can could will would shall should may might has have had not no if so such about over under per""".split())


def stem(w):
    """Crude on purpose: only has to equate fix/fixed/fixes and sample/sampling on both sides."""
    if not w.isalpha() or len(w) < 5:
        return w
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith(("ed", "es")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w[:-1] if w.endswith("e") and len(w) > 4 else w


def terms(s):
    """Index terms for a text: each word as written, its alphanumeric parts, those parts glued together,
    and a stem. 'TQ-40' -> tq-40, tq, 40, tq40; so TQ-40, TQ40 and 'TQ 40' all still overlap."""
    out = []
    for w in re.findall(r"[^\W_][\w.\-]*", s.lower()):
        w = w.rstrip(".-_")
        if not w:
            continue
        out.append(w)
        parts = [p for p in re.split(r"[\W_]+", w) if p]
        if len(parts) > 1:
            out += parts + ["".join(parts)]
        elif (st := stem(w)) != w:
            out.append(st)
    return out


def flat(s):
    """Letters and digits only, lower case: 'Q3 FY27' == 'q3fy27' == 'Q3-FY27'."""
    return re.sub(r"[\W_]+", "", s.casefold())


def toks(s):
    return re.findall(r"[^\W_]+", s.casefold())


def has(text, ans):
    """Is `ans` in `text` as whole tokens? ('50' is not in '350'.) Long answers may differ in spacing:
    'Q3 FY27' still matches 'Q3FY27'."""
    a, t = toks(ans), toks(text)
    if a and any(t[i:i + len(a)] == a for i in range(len(t) - len(a) + 1)):
        return True
    return len(flat(ans)) >= 5 and flat(ans) in flat(text)


# A bare number is what the graders expect ('94', '180'), so drop the unit the model likes to add.
# Single letters (C, V, A, s...) only count as units after a space: '2A' and '12V' may be codes.
UNIT = re.compile(r"^(-?[\d,]*\d(?:\.\d+)?)(?:\s*(?:°\s*[CF]|%|k?W|[kMG]?Hz|ms|[µu]s|ns|secs?|seconds?|mins?|minutes?"
                  r"|hrs?|hours?|days?|[KMGT]i?B(?:/s)?|USD)|\s+(?:[CFVAsmgh]|k?g|[kc]m))$", re.I)


def tidy(ans):
    ans = ans.strip()
    m = UNIT.match(ans)
    return m.group(1) if m else ans


ID = re.compile(r"(?=[\w\-]*\d)(?=[\w\-]*[A-Za-z])[^\W_][\w\-]{2,}[^\W_]")


def ids(text):
    """Identifier-looking tokens (ORR-1847, E7731, ORR-FAN-2214-B): letters and digits mixed."""
    return list(dict.fromkeys(ID.findall(text)))


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.post = defaultdict(dict)       # term -> {chunk: count}; withdrawn documents are left out
        self.dl = []
        for i, c in enumerate(chunks):
            tf = Counter(terms(c["file"] + " " + c["text"]))
            self.dl.append(sum(tf.values()) or 1)
            if c["status"] == "ok":
                for t, n in tf.items():
                    self.post[t][i] = n
        live = [self.dl[i] for i, c in enumerate(chunks) if c["status"] == "ok"]
        self.n = len(live)
        self.avg = sum(live) / len(live) if live else 1.0

    def scores(self, query):
        s = defaultdict(float)
        for t in {t for t in terms(query) if t not in STOP}:
            p = self.post.get(t)
            if not p:
                continue
            idf = math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for i, tf in p.items():
                s[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.dl[i] / self.avg))
        return s

    def search(self, query, k=K, per_file=PER_FILE):
        s, out, per = self.scores(query), [], Counter()
        for i in sorted(s, key=s.get, reverse=True):
            f = self.chunks[i]["file"]
            if per[f] < per_file:
                per[f] += 1
                out.append(i)
                if len(out) == k:
                    break
        return out

    def links(self, ev, question, top=3, rare=6):
        """Follow identifiers out of the best chunks into OTHER files.

        A log line that says 'incident logged against ORR-1847' says nothing about the fix; the bug
        database row for ORR-1847 does. Returns [(chunk_with_the_id, chunk_in_another_file, id)].
        """
        seen = {flat(x) for x in ids(question)}
        out = []
        for a in ev[:top]:
            for x in ids(self.chunks[a]["text"]):
                if flat(x) in seen:
                    continue
                seen.add(flat(x))
                p = self.post.get(x.lower())
                if not p or len(p) > rare:      # nowhere else, or too common to be a link
                    continue
                out += [(a, b, x) for b in p if self.chunks[b]["file"] != self.chunks[a]["file"]]
        out.sort(key=lambda t: len(self.post[t[2].lower()]))    # most specific key first: a ticket id before a product name
        return out


PROMPT = """You answer questions about a company's internal documents. The company and its products are fictional, so use ONLY the numbered evidence below and nothing you remember.

How to answer:
- Give the bare value only, copied exactly as written in the evidence: a number, part number, version, quarter or code. Keep qualifiers that belong to the value (Q3 FY27, REV-C2). No sentence, no explanation, no units.
- Ignore documents that say they are withdrawn, superseded or obsolete. A revision that says it supersedes another is the current one. Prefer the current value over one described as old, previous or "raised from".
- If one record points to another by an identifier (a ticket, part number or error code) and the value lives in the other record, take the value from the other record. If that record is not in the evidence yet, answer NONE and put the identifier in NEED.
- If the evidence does not contain the answer, answer NONE. Never guess.
- SOURCES lists the numbers of the evidence items you needed - every one of them, including an item that only told you which record to look up - and no others.

Evidence:
{evidence}

Question: {question}

Reply with exactly three lines:
ANSWER: <value or NONE>
SOURCES: <numbers separated by commas, or NONE>
NEED: <identifier to look up, or NONE>"""


def render(idx, corpus, question, ev):
    """The prompt, plus the picture files to show alongside it (at most two)."""
    blocks, images = [], []
    for n, i in enumerate(ev, 1):
        c = idx.chunks[i]
        head = f"[{n}] {c['file']} ({c['loc']})"
        if c["kind"] == "image" and len(images) < 2:
            images.append(str(Path(corpus) / c["file"]))
            head += f" - this is Image {len(images)}, attached; text read from it:"
        blocks.append(f"{head}\n{c['text'][:SHOW] or '(none)'}")
    return PROMPT.format(evidence="\n\n".join(blocks), question=question), images


def parse(raw):
    """-> (answer, source numbers, identifier to look up); '' / [] / '' when the model said NONE."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    f = {}
    for k, v in re.findall(r"^\W*(ANSWER|SOURCES|NEED)\W*:\s*(.*)$", raw, re.M | re.I):
        f.setdefault(k.upper(), re.sub(r"^[\s`*\"']+|[\s`*\"']+$", "", v))     # markdown / quotes around the value

    def val(k):
        v = f.get(k, "")
        return "" if v.upper() in ("", "NONE", "N/A", "NULL", "UNKNOWN") else v

    return val("ANSWER"), [int(x) for x in re.findall(r"\d+", val("SOURCES"))], val("NEED")


def answer(idx, llm, corpus, question, deadline=None):
    """-> {answer, citations, confidence, trace}. Refuses ('' and no citations) unless the answer is
    found verbatim in a chunk the model cited."""
    chunk = idx.chunks
    trace = {"rounds": []}
    ev = idx.search(question, K)
    if not ev:
        return {**REFUSE, "trace": trace}
    first = set(ev)
    link = idx.links(ev, question)
    ev += list(dict.fromkeys(b for _, b, _ in link if b not in first))[:EXTRA]
    # a chunk reached ONLY through an identifier: the chunk that held that identifier is a necessary step
    via = {}
    for a, b, _ in link:
        if b not in first:
            via.setdefault(b, a)
    cited, asked = [], set()
    for _ in range(HOPS):
        if deadline and time.monotonic() > deadline:
            break
        prompt, images = render(idx, corpus, question, ev)
        raw = llm.generate(prompt, images, 80)
        ans, nums, need = parse(raw)
        got = [ev[n - 1] for n in nums if 0 < n <= len(ev)]
        cited += [c for c in got if c not in cited]
        trace["rounds"].append({"evidence": [(chunk[i]["file"], chunk[i]["loc"]) for i in ev], "raw": raw})
        if ans:
            return _finish(idx, question, ans, cited, ev, via, trace)
        if not need or flat(need) in asked:
            break
        asked.add(flat(need))
        holder = next((c for c in got + ev if flat(need) in flat(chunk[c]["text"])), None)
        new = [c for c in idx.search(need, 4) if c not in ev]
        if not new:
            break
        if holder is not None:
            if holder not in cited:
                cited.append(holder)
            via.update({c: holder for c in new})
        ev += new
    return {**REFUSE, "trace": trace}


def _finish(idx, question, ans, cited, ev, via, trace):
    chunk = idx.chunks
    ans = tidy(ans)
    holds = [c for c in ev if has(chunk[c]["text"], ans)]       # chunks that contain the value, best-ranked first
    value = next((c for c in holds if c in cited), None)        # prefer one the model said it used
    if value is None and holds:
        value = holds[0]
    if value is None:                                           # not in any text: only fine if read off a picture
        value = next((c for c in cited if chunk[c]["kind"] == "image"), None)
    if value is None:
        trace["why"] = "answer not found verbatim in the evidence"
        return {**REFUSE, "trace": trace}
    files = [chunk[value]["file"]]
    if value in via:                        # we could only reach it through this chunk
        files.append(chunk[via[value]]["file"])
    # A chunk that holds the same value is an alternative source, not a step. A step is a chunk the model
    # cited that shares a lookup key (ticket, part number...) with the value chunk - a key the question
    # did not give and that is not the answer itself.
    mine = {flat(x) for x in ids(chunk[value]["text"])} - {flat(x) for x in ids(question)} - {flat(ans)}
    for c in cited:
        if c not in holds and mine & {flat(x) for x in ids(chunk[c]["text"])}:
            files.append(chunk[c]["file"])
    return {"answer": ans, "citations": list(dict.fromkeys(files)), "confidence": 0.9, "trace": trace}

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
EXTRA = 8       # chunks added by following keys into other files (up to 4 per hop, so first-hop noise cannot starve the second)
HOPS = 3        # model rounds; a round may ask for one more identifier to be looked up
SHOW = 1600     # characters of one chunk the model sees (ingest keeps chunks under this)
HARD = 3        # if the question's own words rank the value chunk below this, the value sits at the end of a chain
SHARED = 3      # a key found in more files than this is a name (a product, a node), not a pointer to one record

REFUSE = {"answer": "", "citations": [], "confidence": 0.0}

# Switches, so prompt variants can be compared on the real model in one run (eval.py --variants).
PROPERTY_RULE = "long"          # "long" | "short" | "off": how hard the prompt pushes "give the final record's property, not the pointer"
LINKED_IMAGES = False           # attach pictures only if the question itself retrieved them. Measured on the real model: an
                                # unrelated picture pulled in by a link made it decline a plain spreadsheet question
RULES = {
    "long": ("- The question asks for one specific property (a date, a version, a number, a code...). Give that property of the record the "
             "chain of pointers ends at, never the pointer itself: if a ticket names a version and the question asks when that version was "
             "released, the answer is the release date written in that version's own release notes. A value belongs only to the record that "
             "carries the matching identifier or version; never take it from a record about a different one.\n"),
    "short": "- Answer with the property the question asks for (a date, a version, a number...) of the record you end up at, not with the identifier that led you there.\n",
    "off": "",
}

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


TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}t\d", re.I)       # 2026-08-11T03:30:00Z is not an identifier


def ids(text):
    """Identifier-looking tokens (ORR-1847, E7731, ORR-FAN-2214-B): letters and digits mixed."""
    return [x for x in dict.fromkeys(ID.findall(text)) if not TIMESTAMP.match(x)]


VERSION = re.compile(r"(?<![\w.])\d+(?:\.\d+){1,3}(?![\w])")

def chunk_keys(text):
    """Lower-case tokens that can point at ONE other record: identifiers and versions. (Not dates: many things happen on a date.)"""
    return {x.lower() for x in ids(text)} | set(VERSION.findall(text))


def is_key(t):
    return bool(VERSION.fullmatch(t) or (ID.fullmatch(t) and not TIMESTAMP.match(t)))


def path_words(file):
    """'engineering/release_notes_5.3.0.txt' -> 'engineering release notes 5.3.0', so '5.3.0' is a whole term."""
    return re.sub(r"[/_]+", " ", file.rsplit(".", 1)[0] if "." in file.rsplit("/", 1)[-1] else file)


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.post = defaultdict(dict)       # term -> {chunk: count}; withdrawn documents are left out
        self.by_file = defaultdict(list)    # file -> its chunks
        self.files_of = defaultdict(set)    # identifier -> the files that mention it
        self.name_files = defaultdict(set)  # key found in a FILE NAME (a version, a date, a node id) -> files
        self.dl = []
        for i, c in enumerate(chunks):
            tf = Counter(terms(c["file"] + " " + path_words(c["file"]) + " " + c["text"]))
            self.dl.append(sum(tf.values()) or 1)
            if c["status"] == "ok":
                self.by_file[c["file"]].append(i)
                for x in ids(c["text"]):
                    self.files_of[flat(x)].add(c["file"])
                for t, n in tf.items():
                    self.post[t][i] = n
        for f in self.by_file:
            for t in set(terms(path_words(f))):
                if is_key(t):
                    self.name_files[t].add(f)
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

    def links(self, ev, question, top=3, rare=6, depth=2, per_hop=4):
        """Follow keys out of the best chunks into OTHER files, up to `depth` hops, `per_hop` new chunks each.

        A log line that says 'incident logged against ORR-1847' says nothing about the fix; the bug database
        row for ORR-1847 does. That row says 'fixed_in: 5.3.0'; the release notes for it live in a file NAMED
        after the version. Two kinds of pointer: an identifier that appears in only a few chunks, and a
        version / identifier that appears in a file name. Returns [(chunk_holding_the_key, chunk_it_leads_to, key)],
        the most specific keys first within each hop.
        """
        score = self.scores(question)
        seen = chunk_keys(question)                     # keys the question already gave are not pointers
        have, frontier, out = set(ev), list(ev[:top]), []
        for _ in range(depth):
            found = []
            for a in frontier:
                fa = self.chunks[a]["file"]
                for t in sorted(chunk_keys(self.chunks[a]["text"])):
                    if t in seen:
                        continue
                    seen.add(t)
                    p = self.post.get(t)
                    if p and len(p) <= rare and ID.fullmatch(t):     # versions lead only through file names
                        found += [(len(p), a, b, t) for b in p if self.chunks[b]["file"] != fa]
                    fs = self.name_files.get(t, ())
                    if 0 < len(fs) <= 2 and fa not in fs:
                        found += [(len(fs), a, max(self.by_file[f], key=lambda c: score.get(c, 0.0)), t) for f in fs]
            found.sort(key=lambda x: x[0])              # a ticket id before a product name
            keep = list(dict.fromkeys(b for _, _, b, _ in found if b not in have))[:per_hop]
            out += [(a, b, t) for _, a, b, t in found if b in keep]
            have.update(keep)
            frontier = keep
            if not frontier:
                break
        return out


PROMPT = """You answer questions about a company's internal documents. The company and its products are fictional, so use ONLY the numbered evidence below and nothing you remember.

How to answer:
- Give the bare value only, copied exactly as written in the evidence: a number, part number, version, quarter or code. Keep qualifiers that belong to the value (Q3 FY27, REV-C2). No sentence, no explanation, no units.
- Ignore documents that say they are withdrawn, superseded or obsolete. A revision that says it supersedes another is the current one. Prefer the current value over one described as old, previous or "raised from".
- If one record points to another by an identifier (a ticket, part number or error code) and the value lives in the other record, take the value from the other record. If that record is not in the evidence yet, answer NONE and put the identifier in NEED.
{property_rule}- If the evidence does not contain the answer, answer NONE. Never guess.
- SOURCES lists the numbers of the evidence items you needed: the item that holds the final value, and every item that only told you which record to look up. No others.

Evidence:
{evidence}

Question: {question}{note}

Reply with exactly three lines:
ANSWER: <value or NONE>
SOURCES: <numbers separated by commas, or NONE>
NEED: <identifier to look up, or NONE>"""


CORRECTION = ("Note: your answer '{ans}' does not appear in the evidence items you cited ({cited}); it appears in item(s) {where}. "
              "If it really came from there, answer again citing that item. If it came from a record about something else, "
              "look again: the value belongs to the record the chain of pointers ends at. If that record is not in the evidence "
              "yet (for example the release notes of a version that a ticket names), answer NONE and put what to look up in NEED.")


def render(idx, corpus, question, ev, note="", first=None, attach=True):
    """The prompt, plus the picture files to show alongside it (at most two). `first` is what the question itself
    retrieved; with LINKED_IMAGES off, a picture that only a link pulled in is described but not attached."""
    blocks, images = [], []
    for n, i in enumerate(ev, 1):
        c = idx.chunks[i]
        head = f"[{n}] {c['file']} ({c['loc']})"
        if attach and c["kind"] == "image" and len(images) < 2 and (LINKED_IMAGES or first is None or i in first):
            images.append(str(Path(corpus) / c["file"]))
            head += f" - this is Image {len(images)}, attached; text read from it:"
        blocks.append(f"{head}\n{c['text'][:SHOW] or '(none)'}")
    return PROMPT.format(evidence="\n\n".join(blocks), question=question, note=f"\n\n{note}" if note else "",
                         property_rule=RULES[PROPERTY_RULE]), images


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


def supported(idx, ans, cited, ev, via):
    """Chunks holding `ans` that the model's citations stand behind: in a file it cited (citations are graded by
    FILE, and a model often names the neighbouring window of the right log), or reached by following a pointer
    out of a chunk it cited (a ticket row that names a version points at that version's release notes, and models
    like to list only the earlier steps). Chunks it cited come first."""
    files = {idx.chunks[c]["file"] for c in cited}

    def anchored(c, seen=()):
        if c in seen:
            return False
        return idx.chunks[c]["file"] in files or any(anchored(a, (*seen, c)) for a in via.get(c, ()))

    hits = [c for c in ev if has(idx.chunks[c]["text"], ans) and anchored(c)]
    return sorted(hits, key=lambda c: c not in cited)


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
    # a chunk reached ONLY through a key: the chunk that held that key is a necessary step. Several chunks may hold the
    # same key (two logs naming one ticket); the one the question's own words favour comes first.
    sc = idx.scores(question)
    via = {}
    for a, b, _ in sorted(link, key=lambda t: -sc.get(t[0], 0.0)):
        if b not in first:
            via.setdefault(b, []).append(a)
    cited, asked, note, corrected, text_only = [], set(), "", False, False
    for _ in range(HOPS + 1):               # +1: the one self-correction below
        if deadline and time.monotonic() > deadline:
            break
        prompt, images = render(idx, corpus, question, ev, note, first, attach=not text_only)
        raw = llm.generate(prompt, images, 80)
        ans, nums, need = parse(raw)
        got = [ev[n - 1] for n in nums if 0 < n <= len(ev)]
        cited += [c for c in got if c not in cited]
        trace["rounds"].append({"evidence": [(chunk[i]["file"], chunk[i]["loc"]) for i in ev], "raw": raw})
        note = ""
        if ans:
            ans = tidy(ans)
            if corrected or supported(idx, ans, cited, ev, via) or any(chunk[c]["kind"] == "image" for c in cited):
                return _finish(idx, question, ans, cited, ev, via, trace)
            # the model answered from something it did not cite: give it one chance to say what it really used
            where = ", ".join(str(n) for n, c in enumerate(ev, 1) if has(chunk[c]["text"], ans))
            if not where:                   # it is nowhere in the evidence: nothing to correct towards
                return _finish(idx, question, ans, cited, ev, via, trace)
            corrected = True
            note = CORRECTION.format(ans=ans, cited=", ".join(map(str, nums)) or "none", where=where)
            continue
        if not need and images and not text_only:
            text_only = True                # a model with a picture in front of it sometimes declines a text question: ask again without it
            continue
        if not need or flat(need) in asked:
            break
        asked.add(flat(need))
        holder = next((c for c in got + ev if flat(need) in flat(chunk[c]["text"])), None)
        new = [c for c in idx.search(f"{need} {question}", 6) if c not in ev]   # the key alone ('5.1.2') matches dozens of rows
        if not new:
            break
        if holder is not None:
            if holder not in cited:
                cited.append(holder)
            via.update({c: [holder] for c in new})
        ev += new
    return {**REFUSE, "trace": trace}


def _finish(idx, question, ans, cited, ev, via, trace):
    chunk = idx.chunks
    holds = [c for c in ev if has(chunk[c]["text"], ans)]       # chunks that contain the value, best-ranked first
    if holds:
        # Models mis-number their sources often, and a refusal scores like a wrong answer, so an answer found verbatim in
        # the evidence is accepted. Which chunk is THE source: one that carries a key the question gave (the ticket id
        # it asked about), then one on a chain (reached by a pointer, or sharing a ticket id with another file), then the
        # one the question's own words rank highest, then one the model cited.
        stand_behind, asked = set(supported(idx, ans, cited, ev, via)), chunk_keys(question)
        skip = {flat(x) for x in ids(question)} | {flat(ans)}
        rare = lambda c: {k for k in ({flat(x) for x in ids(chunk[c]["text"])} - skip) if len(idx.files_of.get(k, ())) <= SHARED}   # noqa: E731
        in_files = defaultdict(set)         # rare key -> files in the evidence that mention it
        for c in ev:
            for k in rare(c):
                in_files[k].add(chunk[c]["file"])
        # "connected": reached through a pointer, or sharing a rare key (a ticket id) with another file's chunk in the evidence
        connected = lambda c: c in via or any(in_files[k] - {chunk[c]["file"]} for k in rare(c))   # noqa: E731
        sc = idx.scores(question)

        def root(c, seen=()):               # the chunk a chain of pointers started from
            srcs = [a for a in via.get(c, ()) if a not in seen]
            return root(srcs[0], (*seen, c)) if srcs else c

        # A row two hops from the log the question names beats a row two hops from some other log: score a chained
        # holder by where its chain STARTS, since the row's own words match the question hardly at all.
        value = min(holds, key=lambda c: (not (chunk_keys(chunk[c]["text"]) & asked), not connected(c), -sc.get(root(c), 0.0),
                                          c not in stand_behind, holds.index(c)))
    else:                                                       # not in any text: only fine if read off a picture
        value = next((c for c in cited if chunk[c]["kind"] == "image"), None)
    if value is None:
        trace["why"] = "the answer does not appear in the evidence"
        return {**REFUSE, "trace": trace}
    chain = [value]                         # every chunk that was a necessary step towards the value
    while chain[-1] in via:                 # we reached it only through the previous one
        srcs = [a for a in via[chain[-1]] if a not in chain]
        if not srcs:
            break
        chain.append(next((a for a in srcs if a in cited), srcs[0]))
    # Other chunks the model cited count as steps only if the value is at the END of a chain: the question's own
    # words do not rank it near the top (so the question alone could not have found it), and the chunk shares a
    # rare key (a ticket, a part number) with the chain that the question did not give and that is not the answer.
    # A chunk that holds the same value is an alternative source, not a step.
    sc = idx.scores(question)
    if sum(1 for v in sc.values() if v > sc.get(value, 0.0)) >= HARD:
        skip = {flat(x) for x in ids(question)} | {flat(ans)}
        grew = True
        while grew:
            grew = False
            keys = {flat(x) for k in chain for x in ids(chunk[k]["text"])} - skip
            keys = {k for k in keys if len(idx.files_of.get(k, ())) <= SHARED}
            for c in cited:
                if c not in chain and c not in holds and keys & {flat(x) for x in ids(chunk[c]["text"])}:
                    chain.append(c)
                    grew = True
    files = [chunk[c]["file"] for c in chain]
    return {"answer": ans, "citations": list(dict.fromkeys(files)), "confidence": 0.9, "trace": trace}

# Mini-Challenge 3: RAG over a mixed document corpus

AMD x lablab.ai AI Academy, Mini-Challenge 3. A container that is pointed at a folder of documents
(pdf, docx, xlsx, csv, txt, log, py, png, jpg), answers questions about them with the **value only**,
and cites the **exact set of files** the answer needed. If the folder does not hold the answer it
returns an empty answer and no citations.

The contract, limits and traps are in the challenge brief; the starter kit's `CONTRACT.md` has them in full.

## How it works

```
app/app.py      thin stdlib client. The harness starts a NEW process per question, so this only
                talks to the resident server and ALWAYS writes a valid output JSON (a refusal on any failure).
app/server.py   resident process (the container's CMD, via start.sh): holds the model and the index
                behind a unix socket, so a question costs seconds instead of a model load.
app/ingest.py   defensive corpus walk: every file in its own try/except + time limit. Parsers for
                pdf / docx (incl. tables) / xlsx (every sheet) / csv / text / images. Tables become one
                chunk per row with the header repeated, so a row still means something when cut out.
                Skips unreadable, encrypted, corrupt and unknown-type files; flags withdrawn documents.
app/rag.py      BM25 over chunks (identifier-aware tokenisation), identifier links between files,
                the prompt, and the answer loop with citation selection.
app/llm.py      Qwen3-VL on the GPU (placed explicitly) + a dev-only ollama backend.
eval.py         scores the way the grader does (normalised answer AND exact citation set).
tests/          offline tests with a scripted fake model; make_corpus.py builds a bigger, nastier corpus.
```

**Answering a question**
1. BM25 top chunks (at most 3 per file; withdrawn documents are never retrieved). File names are search terms too,
   so `5.3.0` finds `release_notes_5.3.0.txt` and `2026-08-11` finds that day's log.
2. *Link following, up to two hops (4 new chunks per hop):* an identifier (`ORR-1847`, a part number, an error code)
   found in the best chunks is looked up in **other files**; a version found in a ticket row leads to the file named
   after it. That is how a log line that only names a ticket reaches the row that holds the fix, and the row that
   names a version reaches that version's release notes, even when the question's own words retrieve neither.
   Dates and timestamps are deliberately not keys, and a key found in many files is a name, not a pointer.
3. The model sees the numbered evidence and replies `ANSWER / SOURCES / NEED`. A picture is attached only if the question
   itself retrieved it (measured on the real model: an unrelated picture pulled in by a link made Qwen3-VL-4B decline a
   plain spreadsheet question), and a refusal with a picture attached is asked once more without it.
   `NEED` asks for one more key to be looked up. If its answer is not in the files it cited, it is told where the
   value does appear and gets one chance to correct itself.
4. **Refusal:** the answer must appear verbatim in the evidence (or be read from a cited picture), otherwise
   `{"answer": "", "citations": []}`. This stops a value that only exists in the encrypted file, or in the model's
   memory, from being returned. Models mis-number their sources often and a refusal scores like a wrong answer, so a
   value found in the evidence is accepted even when the model cited the wrong item.
5. **Which chunk is the source:** one that carries a key the question gave, then one on a chain, then the one the
   question's own words rank highest. A chain's score is where it *starts*: a ticket row matches the question hardly
   at all, but the log it came from does.
6. **Citations by necessity:** the file holding the value, plus every file on the chain of pointers that led to it.
   Other files the model cited count only if the value sits at the END of a chain, meaning the question's own words
   do not rank it near the top (the brief's own test for a real multi-file question), and they share a rare key
   with the chain. A file that merely holds the same value is an alternative source, not a step.

## Develop

```bash
python3 -m venv .venv && .venv/bin/pip install pypdf python-docx openpyxl pillow fpdf2
```

The starter kit's own files are not in this repo. Download `mc3-starter-kit.zip` from the challenge page,
unzip it **into the repo root** (it provides `mc3-corpus/`, `sample-questions.json`, `selfcheck.py`, `CONTRACT.md`)
and run `./setup.sh` once (recreates the empty directory and the `chmod 000` file).

```bash
.venv/bin/python -m unittest discover -s tests -v          # offline: no GPU, no model
.venv/bin/python eval.py --llm ollama --ocr-fixtures dev/ocr_fixtures.json     # local stand-in model (text only)
.venv/bin/python tests/make_corpus.py                      # 55-file corpus with look-alikes -> tests/extra/
.venv/bin/python tests/make_corpus.py --scale 4            # 139 files, many more look-alikes -> tests/extra4/
.venv/bin/python eval.py --llm ollama --corpus tests/extra/corpus --ocr-fixtures tests/extra/ocr_fixtures.json \
                         --questions tests/extra/questions.json tests/extra/questions_para.json
.venv/bin/python eval.py --llm qwen                        # the real model, on an AMD GPU
```

`questions_para.json` asks the same questions in different words; the generated corpora also hold a 3,000-line
log, a three-step chain (log -> ticket -> release notes) and answers that exist only in the encrypted or unreadable file.

### On the AMD hackathon notebook (free GPU, no credit needed)

`notebooks.amd.com/hackathon`, JupyterLab, 3 h/day, 25 GB persistent `/workspace`, **no Docker**, and the egress
proxy may block GitHub, so upload the code as a zip. It is the same base image the container uses.
Long pasted lines get mangled by the browser terminal, so everything is a short `scripts/nb.sh` command:

```bash
bash scripts/nb.sh check      # what can the box reach?
bash scripts/nb.sh install    # deps (torch pinned to the ROCm build) + start the ~9 GB weights download
bash scripts/nb.sh eval       # real model on the starter-kit sample
bash scripts/nb.sh extra      # real model on the generated corpus, questions as written and re-worded
bash scripts/nb.sh big        # same, 139 files
bash scripts/nb.sh cli        # the harness's own flow: resident server + a fresh app.py process per question
```

`--ocr-fixtures` stands in for the vision model when the backend cannot see pictures (dev only).

## Ship

```bash
.venv/bin/pip install huggingface_hub && scripts/get_weights.sh      # ~9 GB into models/ (no network at evaluation)
docker build -t mc3-rag .                                            # base image is ~29 GiB (Docker keeps ~52 GB of it on disk);
                                                                     # the weights' copy peaks at ~4x their size on the way in
scripts/build_image.sh models mc3-rag                                # the same image on a tighter disk (~2x): copies the weights
                                                                     # into a container, deleting each host file as it goes
scripts/docker_check.sh mc3-rag                                      # kit selfcheck + a harness-style run (the real model
                                                                     # needs an AMD GPU; without one, check a stub image)
```

The Dockerfile constrains torch/torchvision to the base image's ROCm build and asserts `+rocm` afterwards
(pip silently swapping in a CUDA torch is one of the documented traps).

Then push the image to a public registry. Docker Hub is far cheaper than GHCR: the base image's layers are mounted from
`rocm/pytorch` instead of uploaded, so only the ~9 GB of new layers travel. The submission goes through the lablab.ai
event page (<https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge>); it showed "Submission deadline Dec 2,
12:30 AM IST" when this was written.

## Status

Validated offline (44 tests): ingestion on the hostile cases (unreadable, encrypted with and without the
`cryptography` package, corrupt, unknown type, empty and locked dirs), retrieval, link following, citation selection,
refusals, prompt switches, and the app/server socket plumbing.

**Validated on an AMD GPU** (Radeon Pro W7900D 48 GB, the hackathon notebook; same base image as the container),
real Qwen3-VL-4B in bf16, pictures read by the model itself:

| | result |
|---|---|
| starter-kit sample, 10 q | **200/200**, <= 1.3 s per question |
| harness-style flow (resident server, a fresh `app.py` per question) | **200/200**; model load 7.8 s, warm-up 3 s, index 10-15 s |
| generated corpus, 55 files, 21 q as written / re-worded | **420/420 / 420/420** |
| generated corpus, 139 files, 21 q as written / re-worded | **420/420 / 420/420** |
| peak VRAM | 9-11 GiB allocated/reserved (limit 1-48) |

How it got there: the first runs on the real model scored 360-400 of 420, and every failure was in chain/citation
logic or prompt sensitivity, not reading. A three-step chain whose last file never reached the model; an extra file
accepted as a "step" because two files share a part number; correct answers refused because the model listed the wrong
item numbers as sources; and `eval.py --variants` showed that an unrelated picture pulled in by a link made the 4B
model decline a plain spreadsheet question (fixed: only pictures the question retrieved are attached). The generated
corpora are my own, so they flatter the pipeline: the graded corpus is different and harder.

**Validated in Docker** (a machine with no AMD GPU, so the model is a stub there):
- `scripts/docker_check.sh` on the real ROCm base: the build, including the guard that fails it if pip swaps the ROCm
  torch for a CUDA one; the kit's `selfcheck.py` (every check passes); `--network none --cap-drop DAC_OVERRIDE`; a
  `chmod 000` file, the encrypted PDF and the unknown type are skipped; a killed server comes back through `start.sh`
  with its index and answers the next question. (`docker cp` runs as you and cannot carry the corpus's `chmod 000`
  file, so the script stages a copy and locks the file inside the container; the kit's own `selfcheck.py` fails at
  step 3 for that reason when run as a normal user.)
- The real image (`scripts/build_image.sh`): base + app + Qwen3-VL-4B weights, 37.6 GiB uncompressed (limit 60 GiB).
  The weights inside it hash-identical to the download; offline inside it, transformers 5.18.0 (the version the
  notebook runs used) builds the architecture (4.44 B parameters; the only name absent from the files is the tied
  `lm_head.weight`), every shape matches, the chat template renders, `MC3_LLM=qwen`, torch is `2.13.0+rocm10.0.0`.

**Not validated:** the real model running inside the real container on an AMD GPU. The notebook ran the same software
and weights without Docker; the machine the image was built on has an NVIDIA GPU, so only the organizers' harness (or an
AMD cloud machine) can do that run.

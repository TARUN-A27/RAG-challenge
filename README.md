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
1. BM25 top chunks (at most 3 per file; withdrawn documents never indexed for retrieval).
2. *Link following:* identifiers (`ORR-1847`, part numbers, error codes) found in the best chunks are
   looked up in **other files**, most specific identifier first. That is how a log line that only names a
   ticket reaches the bug-database row that holds the fix, even when the question's own words don't retrieve it.
3. The model sees the numbered evidence (and up to two attached pictures) and replies
   `ANSWER / SOURCES / NEED`. `NEED` asks for one more identifier to be looked up (up to 3 rounds).
4. **Refusal:** the answer must appear verbatim in the evidence (or be read from a cited picture),
   otherwise `{"answer": "", "citations": []}`. This is what stops a value that only exists in the
   encrypted file (or in the model's memory) from being returned.
5. **Citations by necessity:** the file holding the value, plus files that were a necessary step: the chunk an
   identifier was taken from when the value was reachable only through it, and other chunks the model cited
   that share a lookup key with the value chunk and are not named in the question. Nothing else.

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
.venv/bin/python tests/make_corpus.py                      # 51-file corpus with look-alikes -> tests/extra/
.venv/bin/python eval.py --llm ollama --corpus tests/extra/corpus --questions tests/extra/questions.json \
                         --ocr-fixtures tests/extra/ocr_fixtures.json
.venv/bin/python eval.py --llm qwen                        # the real model, on the AMD GPU
```

`--ocr-fixtures` stands in for the vision model when the backend cannot see pictures (dev only).

## Ship

```bash
.venv/bin/pip install huggingface_hub && scripts/get_weights.sh      # ~9 GB into models/ (no network at evaluation)
docker build -t mc3-rag .                                            # base image is ~29 GiB
python3 selfcheck.py mc3-rag mc3-corpus                              # the harness's own checks
docker run --rm --network none --cap-drop DAC_OVERRIDE ...           # how the harness really runs it
```

The Dockerfile constrains torch/torchvision to the base image's ROCm build and asserts `+rocm` afterwards
(pip silently swapping in a CUDA torch is one of the documented traps).

## Status

Validated offline (27 tests): ingestion on the hostile cases (unreadable, encrypted with and without the
`cryptography` package, corrupt, unknown type, empty and locked dirs), retrieval, link following, citation
selection, refusals, and the app/server socket plumbing.

Pipeline results with a local **text-only stand-in model** (`qwen3:8b` via ollama, stronger than the 4B
vision model that ships), pictures replaced by hand-written transcriptions:

| corpus | score | notes |
|---|---|---|
| starter-kit sample (10 q) | 180/200 | the miss is the pinout diagram: it needs the vision model to look at the picture |
| generated, 51 files (19 q) | 380/380 | two-file chains, withdrawn revisions, near-identical constants, 3 refusals |

The generated corpus is my own, so it flatters the pipeline; the graded one is different and harder.

**Not yet validated:** the Qwen3-VL backend on an AMD GPU (image reading, latency, VRAM), the Docker build
on the mandated base image, and the starter kit's `selfcheck.py` against the built image. Next: run
`eval.py --llm qwen` on an AMD GPU, tune the prompts against the real model, then build and self-check the image.

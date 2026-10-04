#!/bin/bash
# Helper for the AMD hackathon notebook (JupyterLab terminal). Short commands on purpose: long pasted
# lines get mangled by the browser terminal.
#
#   bash scripts/nb.sh check     # what can this box reach?
#   bash scripts/nb.sh install   # python deps (torch pinned to the ROCm build), then start the weights download
#   bash scripts/nb.sh eval      # wait for the weights, then score the real model on the sample corpus
#   bash scripts/nb.sh extra     # the generated 55-file corpus, questions as written and re-worded (the model reads its pictures)
#   bash scripts/nb.sh big       # the same with 139 files: many more look-alikes
#   bash scripts/nb.sh probe     # compare prompt/evidence variants on the real model, one model load (3 min)
#   bash scripts/nb.sh cli       # the harness's own flow: resident server + one app.py process per question
set -u
cd "$(dirname "$0")/.." || exit 1
W=${W:-/workspace}
MODEL=${MC3_MODEL:-$W/models/Qwen3-VL-4B-Instruct}
LOG=$W/download.log
SHOW='^(indexed|#####|===|Q[0-9]+ |score|peak VRAM|    wanted|    asked|   stderr|Traceback|[A-Za-z]*Error)'

# run eval.py with the real model, keep the full trace in $1, show only the summary (and any crash)
run_eval() {
    out=$1; shift
    MC3_MODEL=$MODEL python3 -u eval.py --llm qwen --trace "$@" 2>&1 | tee "$out" | grep -E "$SHOW"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && { echo "eval.py exited with $rc; last lines:"; tail -25 "$out"; }
    echo "(full trace: $out)"
}

case "${1:-}" in
check)
    echo "proxy: $(env | grep -i '^https\?_proxy=' | sed 's|//.*@|//***@|' | head -2 | tr '\n' ' ')"
    for h in pypi.org files.pythonhosted.org huggingface.co cdn-lfs.huggingface.co cas-bridge.xethub.hf.co \
             hf-mirror.com modelscope.cn github.com; do
        printf '%-28s' "$h"
        curl -s -o /dev/null -m 8 -w '%{http_code}\n' "https://$h" 2>/dev/null || echo "blocked"
    done
    df -h "$W" | tail -1
    ;;
install)
    [ -f setup.sh ] && bash setup.sh > /dev/null 2>&1
    pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm)==' > /tmp/constraints.txt
    cat /tmp/constraints.txt
    pip install -q -c /tmp/constraints.txt -r requirements.txt fpdf2 || { echo "pip install FAILED"; exit 1; }
    python3 -c "import torch; assert torch.version.hip, 'torch is no longer the ROCm build'; print('torch', torch.__version__, 'is still ROCm')" || exit 1
    python3 -c "import transformers; from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; print('transformers', transformers.__version__, 'has Qwen3-VL')" || exit 1
    mkdir -p "$(dirname "$MODEL")"
    nohup python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-4B-Instruct', local_dir='$MODEL')
print('DOWNLOAD DONE')" > "$LOG" 2>&1 &
    sleep 20
    echo "weights downloading in the background (about 9 GB). so far: $(du -sh "$MODEL" 2>/dev/null | cut -f1)"
    tail -3 "$LOG"
    ;;
eval|extra|big|probe|cli)
    until grep -q "DOWNLOAD DONE" "$LOG" 2>/dev/null; do
        if ! pgrep -f snapshot_download > /dev/null; then echo "the download is not running. log:"; tail -8 "$LOG"; exit 1; fi
        printf '\rwaiting for the weights: %s   ' "$(du -sh "$MODEL" 2>/dev/null | cut -f1)"
        sleep 5
    done
    case "$1" in
    eval)
        run_eval "$W/run_sample.txt"
        ;;
    extra)
        python3 tests/make_corpus.py > /dev/null || exit 1
        run_eval "$W/run_extra.txt" --corpus tests/extra/corpus --questions tests/extra/questions.json tests/extra/questions_para.json
        ;;
    probe)
        python3 tests/make_corpus.py > /dev/null || exit 1
        run_eval "$W/run_probe.txt" --corpus tests/extra/corpus --brief --questions tests/extra/questions.json tests/extra/questions_para.json \
            --variants cur:PROPERTY_RULE=long noimg:LINKED_IMAGES=False shortprop:PROPERTY_RULE=short,LINKED_IMAGES=False noprop:PROPERTY_RULE=off,LINKED_IMAGES=False
        ;;
    big)
        python3 tests/make_corpus.py --scale 4 > /dev/null || exit 1
        run_eval "$W/run_big.txt" --corpus tests/extra4/corpus --questions tests/extra4/questions.json tests/extra4/questions_para.json
        ;;
    cli)
        # what the harness does: --index once, then a fresh `app.py` process per question
        export MC3_MODEL=$MODEL MC3_INDEX_DIR=${MC3_INDEX_DIR:-$W/idx} MC3_SOCK=${MC3_SOCK:-/tmp/mc3.sock}
        rm -rf "$MC3_INDEX_DIR" "$MC3_SOCK"
        (cd app && exec python3 -u server.py > "$W/server.log" 2>&1) &
        srv=$!
        trap 'kill $srv 2>/dev/null' EXIT
        t0=$SECONDS
        python3 app/app.py --index "$PWD/mc3-corpus" | cut -c1-260
        echo "index pass, including the wait for the model to load: $((SECONDS - t0))s"
        python3 -u eval.py --cli 2>&1 | grep -E "$SHOW"
        echo "--- server startup"; grep -E "model loaded|warmed up|warm-up|ready on" "$W/server.log" | cut -c1-200
        echo "--- server log (tail)"; tail -8 "$W/server.log" | cut -c1-200
        ;;
    esac
    ;;
*)
    sed -n '2,10p' "$0"
    ;;
esac

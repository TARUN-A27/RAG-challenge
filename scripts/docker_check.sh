#!/bin/bash
# Check the container's PLUMBING on a machine with no GPU and no weights.
#   scripts/docker_check.sh          build a stub-model image on the real ROCm base (needs the ~29 GiB base), check it
#   scripts/docker_check.sh IMAGE    check an image that already exists (e.g. a slim Python image built from app/ alone)
# Runs the starter kit's selfcheck, then what the harness does by hand: no network, DAC_OVERRIDE dropped, an unreadable
# file, the server killed mid-run. Does NOT test the model: that is what scripts/nb.sh does on the AMD notebook.
set -e
cd "$(dirname "$0")/.."
IMG=${1:-mc3-rag:stub}
[ -n "$1" ] || docker build --build-arg MC3_LLM=stub -t "$IMG" .

# `docker cp` runs as YOU, so it cannot carry the corpus's chmod-000 file (and the kit's selfcheck, which copies the
# corpus the same way, would fail at step 3). Stage a copy where that file is an empty readable stand-in, and lock it
# again INSIDE the container, where the harness's root has no DAC_OVERRIDE either.
stage=$(mktemp -d)
c=
trap 'rm -rf "$stage"; [ -z "$c" ] || docker rm -f "$c" > /dev/null' EXIT
locked=()
while IFS= read -r f; do locked+=("$f"); done < <(cd mc3-corpus && find . -type f ! -readable -printf '%P\n')
skip=(); for f in "${locked[@]}"; do skip+=(--exclude="$f"); done
tar -C mc3-corpus "${skip[@]}" -cf - . | tar -C "$stage" -xf -
for f in "${locked[@]}"; do : > "$stage/$f"; done

echo "=== the starter kit's selfcheck"
python3 selfcheck.py "$IMG" "$stage"

echo "=== by hand: --network none --cap-drop DAC_OVERRIDE, ${#locked[@]} unreadable file(s), the server killed mid-run"
c=$(docker run -d --network none --cap-drop DAC_OVERRIDE "$IMG")
docker exec "$c" mkdir -p /app/corpus
docker cp "$stage"/. "$c":/app/corpus
for f in "${locked[@]}"; do docker exec -u "$(id -u):$(id -g)" "$c" chmod 000 "/app/corpus/$f"; done

docker exec "$c" python3 /app/app.py --index /app/corpus | LOCKED="${locked[*]}" python3 -c "
import json, os, sys
d = json.load(sys.stdin)
skipped = {s['path']: s.get('why') or s.get('status') for s in d['skipped']}
print('  indexed', d['files'], 'files,', d['chunks'], 'chunks; skipped:')
for p, why in skipped.items(): print('   ', p, '|', str(why)[:80])
missing = [p for p in os.environ['LOCKED'].split() if p not in skipped]
sys.exit('  FAIL: an unreadable file was not skipped: %s' % missing if missing else 0)
"

answered() {    # $1 query id, $2 question: ask in a new process, then check the output file's shape
    local t=$SECONDS
    docker exec "$c" python3 /app/app.py --corpus /app/corpus --query-id "$1" --query "$2"
    docker exec "$c" cat "/app/output/$1_output.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert isinstance(d['answer'], str) and isinstance(d['citations'], list), d
print('  ', '$1', 'ok in $((SECONDS - t))s:', d)"
}
answered query_01 "What is the maximum junction temperature of the TQ-40?"

docker exec "$c" python3 -c "
import os, signal
for p in os.listdir('/proc'):
    if p.isdigit() and int(p) != os.getpid():
        try:
            if b'server.py' in open(f'/proc/{p}/cmdline', 'rb').read(): os.kill(int(p), signal.SIGKILL); print('  killed server pid', p)
        except OSError: pass"
answered query_02 "Which parts were returned under RMA?"       # start.sh must bring the server back with its index

docker exec "$c" python3 -c "import torch; assert torch.version.hip; print('  torch', torch.__version__, 'is still the ROCm build')" 2> /dev/null ||
    echo "  NOTE: no ROCm torch in this image (expected for a slim plumbing image, a failure for the real one)"
echo "=== container checks done"

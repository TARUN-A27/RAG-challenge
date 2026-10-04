#!/bin/bash
# Build the real image (base + app + weights) on a disk too small for `docker build` with the weights in the build context.
# Measured here with a 2 GiB test file: `docker build` peaks at ~3.9x the weights' size (context copy, container layer,
# layer blob, snapshot); create + cp + commit peaks at ~3x. This script also deletes each weights file from the host as
# soon as `docker cp` has put it in the container, which brings the peak down to ~2x. A failed run costs a re-download
# (scripts/get_weights.sh); the container keeps what was copied so far.
#   scripts/build_image.sh WEIGHTS_PARENT_DIR [IMAGE]       e.g. ~/.cache/mc3-weights/models mc3-rag
# Env: MODEL (default Qwen3-VL-4B-Instruct), BASE (image to start from; default: the Dockerfile built with MC3_LLM=stub,
# whose ENV is switched to qwen at the end)
set -e
cd "$(dirname "$0")/.."
SRC=${1:?usage: build_image.sh WEIGHTS_PARENT_DIR [IMAGE]}
IMG=${2:-mc3-rag}
MODEL=${MODEL:-Qwen3-VL-4B-Instruct}
dir="$SRC/$MODEL"
[ -f "$dir/config.json" ] || { echo "no $dir/config.json" >&2; exit 1; }
files=$(cd "$dir" && find . -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort)

need_kb=$(( $(du -sk --exclude=.cache "$dir" | cut -f1) * 22 / 10 + 4194304 ))     # 2.2x the weights + 4 GiB
have_kb=$(df --output=avail -k "$dir" | tail -1)
[ "$have_kb" -ge "$need_kb" ] || { echo "need $((need_kb / 1048576)) GiB free, have $((have_kb / 1048576))" >&2; exit 1; }

echo "=== hashing the weights (compared again inside the final image)"
(cd "$dir" && sha256sum $files) > "$dir.sha256"

base=${BASE:-}
if [ -z "$base" ]; then
    docker build --build-arg MC3_LLM=stub -t "$IMG:nomodel" .
    base="$IMG:nomodel"
fi
c=$(docker create "$base")      # no command override: the image's CMD (start.sh) must survive the commit
empty=$(mktemp -d); mkdir "$empty/$MODEL"; docker cp "$empty/$MODEL" "$c":/models/; rm -r "$empty"

echo "=== copying into container $c (each file leaves the host once it is inside)"
for f in $files; do
    docker cp "$dir/$f" "$c":/models/"$MODEL"/
    rm -f "$dir/$f"
done
docker commit --change 'ENV MC3_LLM=qwen' "$c" "$IMG" > /dev/null

echo "=== verifying the final image"
docker run --rm --network none --entrypoint sh "$IMG" -c "echo MC3_LLM=\$MC3_LLM; cd /models/$MODEL && sha256sum \$(ls -Ap | grep -v /)" > "$dir.verify"
grep -qx 'MC3_LLM=qwen' "$dir.verify" || { echo "FAIL: the image is not set to the real model" >&2; exit 1; }
diff <(LC_ALL=C sort "$dir.sha256") <(grep -v '^MC3_LLM=' "$dir.verify" | LC_ALL=C sort) && echo "weights in the image match the downloaded files"
docker inspect "$IMG" --format 'CMD {{json .Config.Cmd}}'
docker history --human=false --no-trunc --format '{{.Size}}' "$IMG" |
    awk '{s += $1} END {printf "uncompressed layers: %.1f GiB (limit 60)\n", s / 2^30}'
docker rm "$c" > /dev/null
echo "=== done: $IMG"

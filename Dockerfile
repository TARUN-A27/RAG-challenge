# MUST be built on the ROCm base image AMD mandates (checked by layer identity). It ships torch and PIL only.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Trap 2: pip can silently swap the ROCm torch for a CUDA build while resolving some package's torch
# dependency. Constrain torch/torchvision to exactly what the base image has, so a conflict fails the
# BUILD loudly instead of corrupting the image, then assert the result.
COPY requirements.txt /app/requirements.txt
RUN pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm)==' > /tmp/constraints.txt \
 && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /app/requirements.txt \
 && python3 -c "import torch, torchvision; assert '+rocm' in torch.__version__, torch.__version__; print('OK', torch.__version__, torchvision.__version__)"

# The evaluation container has NO outbound network: weights ship inside the image (scripts/get_weights.sh).
# Fail the build, not the evaluation, if they are missing. `--build-arg MC3_LLM=stub` builds a test image with no
# model, only to check the container's plumbing on a machine without a GPU (scripts/docker_check.sh). NEVER submit it.
ARG MC3_LLM=qwen
COPY models/ /models/
RUN [ "$MC3_LLM" = "stub" ] || test -f /models/Qwen3-VL-4B-Instruct/config.json || { echo "weights missing: run scripts/get_weights.sh first" >&2; exit 1; }
ENV MC3_LLM=$MC3_LLM HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

COPY app/ /app/
RUN chmod +x /app/start.sh && mkdir -p /app/corpus /app/output /app/index

# No port, no entrypoint script: the harness execs into the running container. Its job is to keep the
# resident server (model + index) alive.
CMD ["/app/start.sh"]

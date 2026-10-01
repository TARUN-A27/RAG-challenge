"""Model backends. Both expose generate(prompt, images=(), max_new_tokens=64) -> str.

`images` items are file paths or raw bytes. Only the server imports this module.
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.request
from pathlib import Path

MODEL_DIR = os.environ.get("MC3_MODEL", "/models/Qwen3-VL-4B-Instruct")
MAX_SIDE = int(os.environ.get("MC3_MAX_SIDE", "1600"))     # longest image side, bounds the vision tokens

OCR_PROMPT = (
    "Transcribe all text visible in this image, exactly as printed, one item per line. "
    "If it is a diagram, table or label, also write each labelled element as 'label: value' "
    "or 'element: what it is attached to' so that no label is separated from the thing it names. "
    "Write only the transcription."
)


class QwenVL:
    """Qwen3-VL on the GPU, loaded once. Placed on the device explicitly: a CPU run is rejected by the harness."""

    def __init__(self, path=MODEL_DIR):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available() and not os.environ.get("MC3_ALLOW_CPU"):
            raise SystemExit("torch sees no GPU (ROCm builds report as 'cuda'); refusing to run on the CPU")
        self.torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
        self.proc = AutoProcessor.from_pretrained(path)

    @staticmethod
    def _open(x):
        from PIL import Image

        im = Image.open(io.BytesIO(x) if isinstance(x, (bytes, bytearray)) else x)
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        return im

    def generate(self, prompt, images=(), max_new_tokens=64):
        content = [{"type": "image", "image": self._open(i)} for i in images]
        content.append({"type": "text", "text": prompt})
        inputs = self.proc.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class Ollama:
    """DEV ONLY: a local ollama server as a stand-in model. Text only - pictures are ignored."""

    def __init__(self, model=os.environ.get("MC3_OLLAMA", "qwen3:8b")):
        self.model = model

    def generate(self, prompt, images=(), max_new_tokens=64):
        body = {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_predict": max_new_tokens, "num_ctx": 8192}}
        req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r)["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def make(name=None):
    name = name or os.environ.get("MC3_LLM", "qwen")
    return {"qwen": QwenVL, "ollama": Ollama}[name]()


def reader(llm):
    """ocr(bytes) -> text, for ingest: the same resident model reads every picture."""
    return lambda data: llm.generate(OCR_PROMPT, [data], 700)

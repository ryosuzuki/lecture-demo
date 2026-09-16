"""Local Qwen3 reasoning adapter for the Reasoning Lab.

The model is loaded lazily and cached outside the artifact directory. No API key
or hosted inference service is used. Public entry points are ``health`` and
``run`` so the parent server can mount them without a framework dependency.
"""
from __future__ import annotations

import os
import importlib.util
import threading
import time
from typing import Any

MODEL_ID = os.environ.get("REASON_MODEL_ID", "Qwen/Qwen3-0.6B")
CACHE_DIR = os.environ.get("REASON_MODEL_CACHE", "/tmp/openclaw/hf-cache/reasoning-lab")
MAX_TOKENS = 512
MAX_SAMPLES = 16

_lock = threading.RLock()
_model = None
_tokenizer = None
_device = None
_load_error: str | None = None


def _select_device(torch):
    # MPS is dramatically faster on the classroom Mac. CPU remains a complete
    # fallback and is deliberately supported for headless verification.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load() -> tuple[Any, Any, str]:
    global _model, _tokenizer, _device, _load_error
    with _lock:
        if _model is not None:
            return _model, _tokenizer, _device
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            os.makedirs(CACHE_DIR, exist_ok=True)
            device, dtype = _select_device(torch)
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, cache_dir=CACHE_DIR, use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                cache_dir=CACHE_DIR,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()
            _model, _tokenizer, _device = model, tokenizer, device
            _load_error = None
            return model, tokenizer, device
        except Exception as exc:  # surfaced through health and the API wrapper
            _load_error = f"{type(exc).__name__}: {exc}"
            raise


def health(load: bool = False) -> dict[str, Any]:
    """Return adapter status; optionally force the lazy model load."""
    if load and _model is None:
        try:
            _load()
        except Exception:
            pass
    available = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))
    return {
        "reason": available,
        "reason_loaded": _model is not None,
        "reason_model": MODEL_ID,
        "reason_device": _device,
        "reason_error": _load_error,
    }


def _split_output(raw: str, thinking: bool) -> tuple[str, str]:
    """Split only delimiters emitted by Qwen; never manufacture rationale."""
    text = raw.strip()
    if "</think>" in text:
        rationale, final = text.split("</think>", 1)
        return rationale.removeprefix("<think>").strip(), final.strip()
    if text.startswith("<think>"):
        # Generation may have hit its token limit before closing the block.
        return text.removeprefix("<think>").strip(), ""
    return ("", text) if not thinking else (text, "")


def run(
    question: str,
    thinking: bool = True,
    max_tokens: int = 256,
    n: int = 1,
) -> dict[str, Any]:
    """Run measured local inference and return exact generated-token counts."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    max_tokens = max(1, min(int(max_tokens), MAX_TOKENS))
    n = max(1, min(int(n), MAX_SAMPLES))
    model, tokenizer, device = _load()

    messages = [{"role": "user", "content": question.strip()}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(thinking),
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    outputs = []
    run_start = time.perf_counter()

    import torch

    # Sequential calls make every displayed latency an actual measurement.
    # They also allow a future server wrapper to check client cancellation
    # between samples without changing this response schema.
    for index in range(n):
        started = time.perf_counter()
        torch.manual_seed(20260916 + index)
        kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": n > 1,
        }
        if n > 1:
            kwargs.update(temperature=0.8, top_p=0.95)
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
        ids = generated[0, prompt_tokens:]
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        # Remove transport/control tokens only. Thinking delimiters are retained
        # until _split_output so provenance remains explicit.
        for special in (tokenizer.eos_token or "", tokenizer.pad_token or ""):
            if special:
                raw = raw.replace(special, "")
        reasoning, final = _split_output(raw, bool(thinking))
        outputs.append(
            {
                "raw": raw.strip(),
                "reasoning": reasoning,
                "final": final,
                "output_tokens": int(ids.numel()),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )

    return {
        "model": MODEL_ID,
        "device": device,
        "thinking": bool(thinking),
        "prompt_tokens": prompt_tokens,
        "outputs": outputs,
        "total_tokens": sum(item["output_tokens"] for item in outputs),
        "elapsed_ms": round((time.perf_counter() - run_start) * 1000, 1),
    }

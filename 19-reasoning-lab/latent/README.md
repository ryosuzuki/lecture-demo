# Continuous Thoughts — Actual Coconut-Style Inference

This lab displays **real measured model activations**, not invented thinking trajectories.

## What runs

- Public third-party replication checkpoint: `bmarti44/coconut-curriculum-checkpoints`, `coconut/checkpoint_49`.
- Architecture: GPT-2, 124M base parameters, 768-dimensional states, three extra boundary/latent tokens.
- Actual recurrence: run the question + start marker, take the last final-layer hidden state, append it as the next input embedding, repeat, then append the end marker and greedily decode.
- `infer.py` recomputes the prefix for transparency rather than relying on the reference implementation's KV-cache optimization. Learned weights are never updated during this demo.
- The model was trained for synthetic logical questions (ProsQA). It is not a general-purpose reasoning assistant. Its output can be wrong, especially on arbitrary questions.

This checkpoint is NOT an original-author model release. Its model card is a replication/ablation study. No benchmark result is independently reproduced by this demo.

## Modes

1. **Recorded real inference:** three fixed first examples from the original public ProsQA test file; genuine outputs and full vectors, reproducible via `infer.py`. No selection for successful answers. The `recorded-traces.json` provenance includes checkpoint SHA256. Replay works with ordinary static hosting and no GPU.
2. **Live local inference:** run `../server.py` with the Python model environment. Edit the question or number of latent steps and click Run Local Model. CPU inference takes several seconds on the development Mac. The browser submits only to its same-origin localhost server.

## 3D meaning

SVD/PCA is fitted to the states **within the current run**. Axes are neither semantic concepts nor stable across runs. The GUI gives variance retained, full numeric vector inspection, and JSON download. This is an inspectable representation, not a decoded natural-language explanation.

Intermediate unembedding is diagnostic only: these probe tokens are never fed back or claimed to be an actual chain of thought. The answer-token selector shows the actual predictive distribution at each greedy decoding step (the first token is often the `###` answer separator).

## Setup

From repository root:

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r latent/requirements.txt
.venv/bin/python latent/download-model.py
.venv/bin/python server.py
```

`COCONUT_CHECKPOINT` can point at an existing checkpoint. No API key is used. The 807MB model is not bundled. `torch.load` uses `weights_only=True`; downloaded Python reference files are not executed.

Source method and dataset: https://github.com/facebookresearch/coconut
Paper: https://arxiv.org/abs/2412.06769
Model: https://huggingface.co/bmarti44/coconut-curriculum-checkpoints
UI libraries: Three.js 0.170.0 (MIT), vendored for offline replay.

# Learning & Reasoning Lab

[Open the complete lab](https://ryosuzuki.github.io/lecture-demo/19-reasoning-lab/) · [Three training views](https://ryosuzuki.github.io/lecture-demo/19-reasoning-lab/learn/) · [Six operation videos](https://ryosuzuki.github.io/lecture-demo/19-reasoning-lab/walkthroughs/)

## What works on GitHub Pages

- Training (original and three alternatives): actual small neural model, exact gradients and live parameter/probability changes. No GPU needed.
- Search: real MiniLM 384D embeddings computed in the browser, lexical comparison, editable documents, PCA 3D. First use downloads model weights.
- Reasoning: WebLLM runs an open model on supported WebGPU browsers after download. Calculator and Game of 24 work without WebGPU. Generated explanations are not guaranteed faithful introspection.
- Continuous Thoughts: three real recorded Coconut replication runs with full 768D vectors and answer distributions. Interactive replay works everywhere; new-question inference needs the included Python server, not GitHub Pages.
- Six MP4 walkthroughs and English/Japanese teaching notes. Videos depict the local edition; its Python inference option is unavailable on static hosting.

A 3D projection discards dimensions. Search ranking uses full vectors. Training is a deliberately small feedforward language model, not a full Transformer. The Coconut checkpoint is a third-party replication, not an original-author release.

## Optional local inference

From this directory, create a Python 3.12 environment, install `torch transformers numpy` (or `latent/requirements.txt`), run `python latent/download-model.py`, and then `python server.py`. Open http://127.0.0.1:8108/. Qwen weights download on first inference; the Coconut checkpoint is about 807 MB. `COCONUT_CHECKPOINT` and `REASON_MODEL_CACHE` override local model cache locations. Multi-gigabyte weights are not committed.

See TEACHING-GUIDE.md and INSTRUCTOR-JA.md. External browser libraries/models require network access on first use.

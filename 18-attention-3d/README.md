# Attention in 3D

Open `3d/index.html` through a static HTTP server. All runtime dependencies and the preset GPT-2 embedding subset are bundled; no API key is needed.

New input lesson: tokenization → embedding lookup → position addition → existing attention lesson. All authored teaching content is English.

GPT-2-small vectors: 768D, vocabulary 50,257. `3d/gpt2-input.json` contains 25 actual learned token vectors and 16 position vectors extracted from the official safetensors weights with HTTP byte ranges. Stored tensors are transposed visually to match 3Blue1Brown's column convention. The 3D arrows display components 0–2, not a full semantic projection. Tokenizer: gpt-tokenizer 4.0.0, r50k_base.

Custom tokenizer inputs work locally up to 16 tokens; custom IDs outside the bundled weight subset are explicitly unsupported in the embedding view. No weights are invented. Existing attention scenes are illustrative and are not an end-to-end GPT-2 execution.

See `3d/input-guide.html`, `3d/input-verification.html`, and `provenance.html`. Original reconstruction attribution/license files are preserved. GPT-2 and gpt-tokenizer have MIT licenses, included separately.

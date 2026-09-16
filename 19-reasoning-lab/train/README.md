# Next-Token Training Lab

An English, projector-readable interactive demo for a recap of next-token training. It is a real small neural language model, not a scripted animation and not a Transformer.

## Run

```bash
cd stuff/2026/2026-09-16/reasoning-lab/train
python3 -m http.server 8765
```

Open `http://localhost:8765/` in a browser. Internet access is needed only for pinned Three.js `0.167.1` modules from jsDelivr.

## Model and math

The corpus is an inspectable list of 18 two-token-context training pairs in `model.js`. Vocabulary size is derived from those pairs. Given context tokens `x₁, x₂`:

1. Look up both learned 3D embeddings and average them: `e = (E[x₁] + E[x₂]) / 2`.
2. Compute a 4D hidden state `h = tanh(eW + b)`.
3. Compute vocabulary logits `z = hO + c`.
4. Compute `p = softmax(z)` and cross-entropy `L = -log p[target]`.
5. Backpropagate exact analytic gradients through softmax, output projection, tanh, hidden weights, and both selected embedding rows.
6. SGD updates every parameter by `weight -= learning_rate * gradient`.

The 3D network is a visualization of this exact `3 → 4 → vocabulary` neural model. Edge colors encode weight sign, opacity encodes magnitude, and node scale uses current activations/probabilities. Clicking an edge selects that actual matrix element. The slider mutates it and immediately reruns the real forward pass.

The optional loss surface recomputes the full forward pass over a grid that varies two distinct real weights. It is a 2D slice of a higher-dimensional objective, not the whole loss landscape. It intentionally stays separate from the right-side probability distribution and is recomputed whenever visible weights or training state change.

## Pedagogy

Use the numbered flow:

1. Predict with fixed weights.
2. Reveal the corpus target.
3. Compute `-log P(target)`.
4. Show the gradient on the selected weight.
5. Update all weights by SGD.
6. Predict again and compare the distribution.

`Train 100` cycles through the real sample corpus. Save/restore is an in-memory checkpoint. Reset uses the same seeded initialization for repeatable classroom instruction.

## Verification

Run:

```bash
node tests/model.test.mjs
```

The test checks probability normalization, cross-entropy identity, five finite-difference gradient checks (embedding, hidden matrix, bias, output matrix, output bias), meaningful loss decrease, and checkpoint restoration.

Browser verification should cover orbiting, clicking a weight, live slider updates, the six-stage lesson flow, one/100 training steps, reset/checkpoint, responsive layout, and the loss surface.

## Limitations

- This is an honestly labeled tiny feed-forward neural language model with a two-token context. It is not a Transformer, does not implement attention, and is not an LLM.
- The tiny corpus is designed for transparent classroom calculation, not linguistic coverage.
- A 3D embedding dimension is used so every displayed embedding component is an actual learned number; it is not PCA or a claimed semantic axis.
- The loss surface varies only two weights while holding all others fixed.
- Training one pair can worsen another. `Train 100` uses a deterministic pair order rather than random minibatches.
- There is no backend and no learner data leaves the browser.

## Screenshots

Generated verification screenshots are stored in `screenshots/`.

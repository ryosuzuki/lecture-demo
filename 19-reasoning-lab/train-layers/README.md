# Training Through the Layers

An English, projector-ready interactive lesson. It is a real tiny **two-context-token feed-forward neural language model**, not a Transformer and not a simulation of hidden chain-of-thought.

## Model

For context tokens `x1,x2`, the model averages their learned 3D embeddings, then computes `h=tanh(eW+b)`, `logits=hO+c`, and softmax probabilities over the actual 19-token vocabulary. Loss is `-log P(target)`. Backpropagation and SGD update the matrices shown in the 3D scene. The corpus includes simple controlled pairs plus `their final → project` and `update the → weights`.

## UI

- Left: full natural sentence, with the exact two-token context highlighted; actual input and hidden vectors.
- Center: orbitable numeric matrix slabs. Every clickable tile reads/writes an actual parameter.
- Right: actual softmax distribution, target probability, cross-entropy, derivative, and persistent loss history.
- Workflow: Predict → Reveal Target → Measure Loss → Backpropagate → Update All Layers.

## Run and verify

Serve this directory with any static server. Three.js 0.167.1 is pinned from jsDelivr. No model/API/network service is used after libraries load.

```sh
node tests/model.test.mjs
python3 tests/browser_check.py
```

The finite-difference checks cover all parameter families. Browser verification exercises selection, manual perturbation, staged gradient/update, 20 SGD updates, reset, and screenshot capture.

## Limitations

This intentionally small model sees only the final two context tokens. It has no attention, Transformer blocks, tokenizer, or private reasoning trace. The full sentence supplies classroom context but is visually separated from the exact model input.

Artifacts: `screenshots/layers-after-training.png`, `video/training-through-layers-walkthrough.mp4`, and `TEACHING-SCRIPT.md`.

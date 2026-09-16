# 20 · Inside a Learning Transformer

One numerical model powers the input vectors, layered 3D activations, attention, next-token probabilities, loss, and parameter updates. No iframe and no illustrative replacement activations.

## Model

304 trainable scalar parameters. 4-dimensional embeddings, learned position vectors, one causal scaled dot-product attention head, output projection and residual, 8-unit tanh MLP with residual, unembedding and softmax. Omits layer normalization/dropout/multiple blocks to keep every operation inspectable. Uses a tiny authored word-token corpus, **not** pretrained LLM weights or a BPE tokenizer. Full displayed context (up to8 tokens) is processed.

One-example training demonstrates optimization/memorization, not general language competence. Right-hand loss is exactly `-ln(P(observed next token))`. Inference recomputes activations with fixed parameters. Gradient calculation alone does not change parameters.

## Teaching sequence

1. Predict: ask what token the untrained model favors.
2. Expand: follow a selected token through embedding, attention, and MLP. Layer numbers are activations, not trainable weights.
3. Change a parameter: keep the sentence fixed and watch the actual output distribution change.
4. Reveal the target; relate its probability to negative log loss.
5. Inspect the gradient, then apply an update. Compare actual before/after values.
6. Loss landscape: only the chosen two parameters move, all others stay fixed. Height is this same example's loss, not a decorative bowl.
7. Change the sentence without training: activations change but weights do not. Transition to inference/reasoning.

## Verification

`node --test model.test.mjs` checks nine parameter groups with finite differences, causal masking, normalized probabilities, exact loss correspondence, reset, convergence on one example, and loss-slice state restoration/coordinate-only updates.

Uses shared Three.js/OrbitControls from ../18-attention-3d/vendor. Serve from repository root; no backend or model downloads required. Previous18 and19 demos are preserved.

## Sources

Teaching design informed by the official3Blue1Brown text adaptations of mini-llm, GPT, attention, gradient descent, and backpropagation. See DESIGN.md for sources and review scope. Source review is not a claim of complete video playback. Source links remain in documentation, not the classroom interface.

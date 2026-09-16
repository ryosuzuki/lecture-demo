# 20: Source-Grounded Teaching and Visual Audit

## Verified scope
Read official lesson text adaptations via fresh HTTP200 downloads on 2026-09-16; saved text in `/tmp/openclaw/3b1b-twenty/`. This is article reading, **not video playback** or frame-accurate scene review. No new video timestamp claims.

- https://www.3blue1brown.com/lessons/mini-llm — distribution over next tokens, tunable dials, held-out final token, training versus generation, attention/MLP overview.
- https://www.3blue1brown.com/lessons/gpt — tokens→embedding→attention/MLP blocks→last vector→unembedding→softmax; distinguishes trained weights from input-dependent data.
- https://www.3blue1brown.com/lessons/attention — actual Q/K scores, softmax, causal mask, weighted V updates, MLP operates independently at each position.
- https://www.3blue1brown.com/lessons/gradient-descent — connect wrong output to scalar objective BEFORE introducing surface; one/two parameter analogies; negative gradient and learning rate; local, not guaranteed global improvement.
- https://www.3blue1brown.com/lessons/backpropagation — sensitivity per weight, chain-rule propagation backward, aggregating many examples versus one example, minibatch approximation.

Inspected `18-attention-3d/3d/index.html` and `app.js`. Preserve its visual language: big black 3D stage, token-column bundles through depth, expanded layer planes, token selection, oblique/front/orbit/reset, concise inspection sidebar. The old pipeline synthesizes some activations with `Math.sin`; do NOT reuse these numerical values or treat its scene as actual trained Transformer output. Preserve rendering style, bind it to a new actual trace.

## One coherent 8-minute sequence

1. **Predict (1 min).** Left: one short complete training sentence with final token hidden; display the actual input token IDs and embeddings. Center: collapsed layered transformer. Right: computed probability distribution. Ask students which continuation it currently favors. Do not start with a wall of matrices.
2. **Inside the Transformer (2 min).** Expand the planes from Embedding → Attention → MLP → Unembedding. Each vertical column is the SAME token position across depth. Click a token and inspect its actual vector at each stage. Attention links cross positions; MLP operates within a position. Last position's final vector feeds right-hand next-token probabilities. Forward pass does not train.
3. **One Dial (1 min).** Click a stage to expose its actual trainable matrix; select one parameter. Scrub its number while recomputing the entire forward pass. Highlight changed activations and changed probabilities. Freeze the sentence so causality is visible. Keep matrix editing in flat readable panel rather than rotated dense text.
4. **Reveal Target and Measure Error (1 min).** Reveal observed final token. Its probability remains highlighted right. Show `L = -ln(p_target)` with the same actual number substituted; e.g. p=.10 → L=2.303. This is the loss of THIS training example, not general accuracy or dataset average. Target is from data, not the model's argmax.
5. **Backpropagate (1 min).** Reveal numeric gradient on selected weight and reverse direction through stages. Weight currently `w`, sensitivity `dL/dw`, learning rate `eta`, update `w_new = w - eta*dL/dw`. Distinguish calculating gradient from applying update. Animation is an explanation of calculation order, not a fabricated trajectory.
6. **Update and Predict Again (1 min).** Apply actual parameter updates; flash before/after numbers; rerun same input. Show target probability and loss before/after. Actual history only. If large learning rate makes loss rise, show that honestly; no pre-drawn descending curve.
7. **Why Downhill? (1 min).** Optional tab/overlay shows actual two-weight loss surface with other weights held fixed. Horizontal axes are named parameters, vertical is same example loss, current sphere, negative-gradient direction and real update path. Updating all weights changes the surface: regenerate or clear path; never show old landscape as fixed truth.
8. **Inference handoff.** Freeze weights and change prompt/sample a token. Explain that activations change without further weight learning. This leads into today's reasoning lecture without conflating longer inference with gradient training.

## Visual hierarchy and interaction requirements

- Left = sentence/input vectors; center = dominant 3D layer architecture; right = output probabilities, target and loss. Keep entire causal path simultaneously visible.
- Use gray for activations and a distinct blue/red signed color scale for parameters, as the official GPT lesson does. Label both explicitly; users often confuse them.
- Layer click inspects the layer's true values and optionally its parameter matrix. Numerical cell click locks a stable weight ID. Preserve selected ID through training, not a changing sorted-list index.
- Big readable layer labels, sparse visible vector numbers, full vectors in inspector; don't put hundreds of tiny labels into rotating depth planes.
- Highlight update magnitude with a temporary color pulse while retaining actual signed numeric values. Do not animate unrelated weights arbitrarily.
- Right bars update in place with stable vocabulary ordering or provide explicit ranked mode. Otherwise moving rank can obscure probability change.
- Default scene has one actual tiny causal Transformer block if that is implementation scope; do not draw two Attention+MLP blocks when model computes one. Mark layer norm/residual simplifications in About, not long student-facing caveat paragraphs.
- Expand changes geometry only; it must not change model values. Orbit is optional; front view should remain usable on mobile.
- Keep 3B1B references in instructor/source documentation, not the live lecture screen, per user request.

## Numerical/semantic acceptance checks

- Same model trace supplies every visible stage, selected values, right logits/softmax, and loss.
- Attention rows sum to1 (or columns if chosen convention); future positions masked; labels state query/source orientation.
- Softmax probabilities sum to1. Logits are not probabilities.
- Analytic selected gradient agrees with central finite differences within numerical tolerance.
- Weight edits recompute activations/probabilities/loss immediately; reset deterministically restores all weights.
- Manual edits invalidate or restart learning history; target change likewise.
- One-example repeated training is labelled as such. An optional corpus/batch mode must average actual example losses, not relabel one-example loss.
- Two-parameter surface reuses exact model loss. It need not be a perfect bowl. Do not fabricate a picturesque basin.
- No assertion that tiny model understands natural language generally or that every training step monotonically lowers every example loss.

## Recommendation
Build **one integrated20** instead of stitching the old illustrative attention iframe beside an unrelated MLP. Use18's excellent layered spatial architecture, but let one actual tiny Transformer drive prediction, inspection, weight edits and gradient updates. This satisfies the user's educational/visual request and fixes19's disconnected model metaphors.

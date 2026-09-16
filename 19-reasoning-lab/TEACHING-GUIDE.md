# From Learning to Reasoning — September 16, 2026

English student-facing UI. Four independent labs in one state-preserving shell.

## Before class

Open `https://ryosuzuki.github.io/lecture-demo/19-reasoning-lab/`, or `http://127.0.0.1:8108/` with the optional Python server. Select each tab once to load its libraries and model weights. Keep this tab open. Browser embedding search needs an initial model download. GitHub Pages uses browser WebGPU reasoning and recorded Coconut replay; new Coconut inference needs the supplied Python server and model environment. Stored continuous-state traces are actual model runs, not canned explanatory fiction.

The local host address refers to THIS computer; it is not a public or student-accessible website. Start `server.py` on a different machine after copying the bundle to use it there. Opening the HTML directly with `file://` is not supported.

## 1. Learn (10–12 minutes)

Ask: **What changes when a model learns?**

- Left is the real training sentence and actual input vectors. Center is an inspectable trainable neural model. Right is always its next-token distribution.
- Predict before revealing the target. A target is observed training data, not the model's favorite word.
- Cross-entropy is `−ln P(target)`: if the correct token has 0.1 probability, loss is about 2.303; at 0.5, loss is about 0.693. The loss is a scalar assessment of these parameters on this example, not another output word.
- Click an actual weight and move its slider. Notice that multiple probabilities change: softmax renormalizes them.
- Reveal the selected weight's gradient. It is the local sensitivity of loss to that parameter; it is not the weight itself.
- Take one SGD step. Then Train 100. Compare distributions and change to another example.
- Open the two-weight loss surface. Every point is a forward calculation using those two weights, all others fixed. Height is loss; horizontal axes are actual parameter values. This is not the model's entire high-dimensional landscape.
- Reset restores a reproducible starting point. Different examples can compete, and larger learning rates are not always better.

This deliberately small neural language model is NOT a full Transformer. Its exact dimensions/context are shown in the lab. The simplicity makes all displayed quantities genuinely calculable. Real Transformer activations are available in the Continuous Thoughts tab.

## 2. Search (8 minutes)

Ask: **Can a document be relevant without sharing the exact words?**

1. Search for Sony headphones plus express shipping. The actual product and shipping chunks have different roles.
2. Switch semantic/cosine and lexical/BM25. A word-overlap method and a learned-embedding method use different signals.
3. Rotate the point cloud and click the query/documents to inspect vectors. Rankings use 384D vectors, not distances in the 3D drawing.
4. Edit a shipping price and re-embed. The document knowledge changes without training the embedding model.
5. Change top-k and discuss what evidence an answering model would receive. This tab is a retrieval lab, not a claim that a downstream answer model has already been executed.

## 3. Reason & Tools (10–15 minutes)

Ask: **How do we distinguish a plausible explanation from a correct answer?**

- Use the actual model mode to compare direct output and generated reasoning. Generated reasoning text is observable output, not a complete transparent account of internal computation.
- A local model can be wrong or hit its generation limit. Keep those outcomes visible; do not replace them with a scripted success.
- Vary sample count and compare token/time costs against a deterministic correctness check. Multiple agreeing answers need not be correct.
- Walk through model → application → calculator → returned result. The calculator performs exact operations; this illustrative handoff is labeled separately from actual model-generated tool calls.
- Game of 24 uses exact rational arithmetic. Try `(13−9)×(10−4)=24`. Expand/check/backtrack with different search budgets. This explicit search program is not the hidden reasoning of the chat model.

## 4. Continuous Thoughts (5–8 minutes)

Ask: **Must every intermediate computation become a word?**

- Start with a recorded real run of the public third-party Coconut replication checkpoint. Step through actual 768D states. Click a point and inspect numerical components.
- The preceding hidden state is fed back as an input embedding, without decoding a word between steps.
- Inspect the generated answer tokens on the right. Intermediate token probes are diagnostic only, not words that were actually produced.
- Run a new question or change the latent-step count using the local inference server. This checkpoint is specialized for synthetic logical questions; arbitrary prompts may fail.
- PCA is fitted separately for each run; do not compare geometric paths between runs as if the axes were semantic concepts. A 3D path is not an explanation of what the model 'thought.'

## Video links used for design

- 3Blue1Brown, **Large Language Models Explained Briefly**, [0:41](https://www.youtube.com/watch?v=LPZh9BOjkQs&t=41s): prediction distributions and many parameter dials.
- **Gradient Descent, How Neural Networks Learn**, [3:01](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=181s): cost/loss; [6:55](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=415s): descent; [11:18](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=678s): vector of updates.
- [Official illustrated LLM lesson](https://www.3blue1brown.com/lessons/mini-llm/) and [gradient descent lesson](https://www.3blue1brown.com/lessons/gradient-descent/).

Verification scope: official written explanations and published chapter times. Not frame-exact video edit points or a claim of complete video viewing.

Single-head extension, September 9 2026

Order: text/embeddings → embedding directions → queries → full single-head calculation → multi-head → full model.

Use lesson-flow.html with slides 63–75. Numbers in single-head.js are deliberately chosen 3D teaching values, not GPT weights; all matrix products, masking, scaled softmax, weighted values and residual addition are computed. Existing real GPT-2 input weights and GloVe vectors remain separate.

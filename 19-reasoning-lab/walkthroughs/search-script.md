# Search walkthrough — instructor script

**Teacher question:** Can a document be relevant without sharing the exact words in the query?

1. Read the Sony headphones + express-shipping query and ask students to predict which two chunks are needed.
2. Run the real MiniLM search. Explain that ranking uses cosine similarity over the original 384D vectors.
3. Open a result and identify source text, cosine score, PCA coordinates, and vector components.
4. Switch to lexical BM25. Ask: *Which ranking changed, and which words caused it?*
5. Set top-k to 2. Ask: *What evidence would an answering model now lose?*

**Teaching boundary:** The 3D PCA view distorts distances; it does not perform ranking. This is retrieval, not a generated answer.


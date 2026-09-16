# Embedding Search Lab

## Run

Serve this folder over HTTP (ES modules do not reliably run from `file://`):

```sh
python3 -m http.server 8765
```

Open `http://localhost:8765/` in a current Chrome, Edge, Firefox, or Safari browser. Internet access is required on first use for Plotly, Transformers.js, and `Xenova/all-MiniLM-L6-v2`. Model files are cached by the browser when permitted. A model-download failure is shown as an error; the page never substitutes fake semantic scores.

## 8-minute teaching sequence

1. Ask students to predict whether one chunk can answer the default product-plus-shipping question.
2. Search and point out that the product and shipping chunks occupy ranks 1 and 2.
3. Switch to **Lexical BM25**. Compare rank changes without implying that semantic is always better.
4. Rotate the 3D plot, click a document and the gold query, and inspect their vector values.
5. Emphasize the blue warning: ranking uses 384D cosine; PCA is only the 3D map.
6. Select Shipping Information, change `$12.99` to `$19.99`, and click **Update + re-embed**. The model is not retrained; one index vector and the PCA display are recomputed.

## Verified behavior (2026-09-16)

- Real browser inference loaded `Xenova/all-MiniLM-L6-v2` and produced finite 384D vectors for 20 source chunks.
- The default query ranked the Sony product first (`0.659151`) and shipping information second (`0.557694`) by original-vector cosine similarity.
- Corpus PCA and out-of-sample query projection produced finite 3D coordinates.
- The 3D Plotly view created two traces (documents plus query) in WebGL and was captured in `verified-screenshot.png`.
- Editing the shipping chunk triggered a fresh embedding; the modified chunk remained semantic rank 2 and its new vector was finite.

## Honest boundaries

- The page embeds a selected 20-chunk subset copied exactly from the cited e-commerce teaching dataset; it does not fetch or silently change the corpus at runtime.
- PCA preserves the largest linear variation, not exact semantic neighborhoods. The 3D query point is projected with the corpus mean and fitted components.
- BM25 is a transparent lexical baseline implemented in the page. It is not labeled semantic.
- Added/edited documents live only in memory and reset on reload. This is intentional for a portable classroom demo.
- There is no generated answer: the lab teaches retrieval and source inspection without pretending a canned answer is model generation.

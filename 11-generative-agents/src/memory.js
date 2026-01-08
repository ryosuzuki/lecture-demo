// Memory stream (simplified).
// - Each memory is a natural language sentence + timestamp + importance score.
// - Retrieval score = relevance + recency + importance (in the paper, they combine these signals).
//   We approximate relevance with TF-IDF cosine similarity (instead of embedding vectors).

import { uniq } from "./util.js";

const STOPWORDS = new Set([
  "the","a","an","and","or","to","of","in","on","for","with","at","from","by","as","is","are","was","were","be",
  "i","you","he","she","they","we","me","my","your","his","her","their","our",
  "this","that","it","its","there","here","then","so","but",
]);

function tokenize(text) {
  // very light tokenizer: lower, remove punctuation, split whitespace.
  const cleaned = text
    .toLowerCase()
    .replace(/[\u3000]/g, " ")
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!cleaned) return [];
  const raw = cleaned.split(" ");
  return raw.filter(t => t && !STOPWORDS.has(t) && t.length >= 2);
}

function cosineSparse(vecA, vecB) {
  // vec: Map token->weight
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (const [k, wa] of vecA) {
    normA += wa * wa;
    const wb = vecB.get(k);
    if (wb != null) dot += wa * wb;
  }
  for (const [, wb] of vecB) normB += wb * wb;

  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class MemoryRecord {
  constructor({ id, time, text, importance = 3, type = "observation" }) {
    this.id = id;
    this.time = time; // Date
    this.text = text;
    this.importance = importance; // 1..10
    this.type = type;

    this.tokens = tokenize(text);
    this.tf = new Map(); // token -> tf
    for (const t of this.tokens) this.tf.set(t, (this.tf.get(t) ?? 0) + 1);
  }
}

export class MemoryStream {
  constructor() {
    this.records = [];
    this._df = new Map(); // token -> doc frequency
  }

  add(record) {
    this.records.push(record);
    const unique = uniq(record.tokens);
    for (const t of unique) this._df.set(t, (this._df.get(t) ?? 0) + 1);
  }

  _idf(token) {
    const N = this.records.length || 1;
    const df = this._df.get(token) ?? 0;
    // smooth IDF
    return Math.log((N + 1) / (df + 1)) + 1;
  }

  _tfidf(tfMap) {
    const out = new Map();
    for (const [t, tf] of tfMap) out.set(t, tf * this._idf(t));
    return out;
  }

  retrieve(query, now, k = 8, { types = null } = {}) {
    const qTokens = tokenize(query);
    const qTF = new Map();
    for (const t of qTokens) qTF.set(t, (qTF.get(t) ?? 0) + 1);
    const qVec = this._tfidf(qTF);

    const scored = [];
    for (const rec of this.records) {
      if (types && !types.includes(rec.type)) continue;

      // relevance: cosine(TF-IDF(query), TF-IDF(memory))
      const mVec = this._tfidf(rec.tf);
      const relevance = cosineSparse(qVec, mVec);

      // recency: exp decay by hours (paper uses 0.99^Δt hours)
      const hours = Math.max(0, (now.getTime() - rec.time.getTime()) / 3600000);
      const recency = Math.pow(0.99, hours);

      // importance normalized 0..1
      const importance = Math.max(1, Math.min(10, rec.importance)) / 10;

      const score = relevance + recency + importance;
      scored.push({ rec, score, relevance, recency, importance });
    }

    scored.sort((a,b) => b.score - a.score);
    return scored.slice(0, k);
  }

  recent(n = 10) {
    return this.records.slice(-n);
  }
}

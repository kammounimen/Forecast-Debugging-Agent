# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05d - Hybrid RAG: BM25 + multilingual embeddings + RRF fusion
# MAGIC Module notebook, loaded from other notebooks with `%run ./05d_rag_hybrid`.
# MAGIC
# MAGIC **Prerequisite in the CALLING notebook** (not here, because `restartPython()` would wipe the state):
# MAGIC ```
# MAGIC %pip install rank-bm25 sentence-transformers --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install rank-bm25 sentence-transformers --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re, json, unicodedata
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Run first, in the calling notebook: %pip install rank-bm25 sentence-transformers "
                      "then dbutils.library.restartPython()") from e

_catalog = spark.sql("SELECT current_catalog()").first()[0]
RAG_TABLE = f"{_catalog}.forecast_debug.rag_documents"      # only table read by this module
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # multilingual (French included)
RRF_K = 60


def _chunk_document(doc_id, title, content, min_body_chars=40):
    """One passage per heading (#, ##, ###). Sections with no body (title only) are skipped."""
    chunks, section, body = [], title, []

    def flush():
        text = "\n".join(body).strip()
        if len(text) >= min_body_chars:
            chunks.append({"doc_id": doc_id, "title": title, "section": section, "body": text})

    for line in content.splitlines():
        if re.match(r"#{1,3} ", line):
            flush()
            body, section = [], line.lstrip("# ").strip()
        else:
            body.append(line)
    flush()
    return chunks


def _tokenize(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9_]+", s)

# COMMAND ----------

# Build passages and both indexes
CHUNKS = []
for d in spark.table(RAG_TABLE).collect():
    CHUNKS.extend(_chunk_document(d["doc_id"], d["title"], d["content"]))
for i, c in enumerate(CHUNKS):
    c["chunk_id"] = f"{c['doc_id']}#{i}"
    c["text"] = f"{c['title']}\n{c['section']}\n{c['body']}"      # indexed text

_BM25 = BM25Okapi([_tokenize(c["text"]) for c in CHUNKS])
_MODEL = SentenceTransformer(EMBEDDING_MODEL)
_EMB = _MODEL.encode([c["text"] for c in CHUNKS], normalize_embeddings=True)


def rank_bm25(query):
    return list(np.argsort(-_BM25.get_scores(_tokenize(query))))


def rank_dense(query):
    q = _MODEL.encode([query], normalize_embeddings=True)[0]
    return list(np.argsort(-(_EMB @ q)))


def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion. Returns a list of (passage_index, score)."""
    score = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)

# COMMAND ----------

# Tool exposed to the agent
def search_knowledge(query: str, k: int = 3) -> dict:
    """Hybrid search in the documentation (rules, definitions, incidents, runbook)."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "query vide"}
    try:
        k = max(1, min(int(k), 5))
    except (TypeError, ValueError):
        k = 3
    fused = rrf([rank_bm25(query), rank_dense(query)])[:k]
    return {"query": query, "results": [
        {"doc_id": CHUNKS[i]["doc_id"], "section": CHUNKS[i]["section"],
         "score": round(score, 4),            # RRF fusion score, not a cosine similarity
         "text": CHUNKS[i]["body"]} for i, score in fused]}


SEARCH_KNOWLEDGE_SCHEMA = {"type": "function", "function": {
    "name": "search_knowledge",
    "description": ("Cherche dans la documentation interne : règles de fraîcheur, tolérance d'écart, "
                    "définitions des features, versions de modèles, incidents passés et runbook. "
                    "À utiliser pour interpréter les faits observés."),
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "Question ou mots-clés, en français"},
                                  "k": {"type": "integer", "description": "Nombre de passages (1 à 5)", "default": 3}},
                   "required": ["query"]}}}


def evaluate_retrievers(tests, verbose=True):
    """tests: list of (query, {expected doc_ids}). Returns recall@3, top-1 and MRR per retriever."""
    rankers = {"BM25": rank_bm25, "Dense": rank_dense,
               "Hybride": lambda q: [i for i, _ in rrf([rank_bm25(q), rank_dense(q)])]}
    out = {}
    for name, fn in rankers.items():
        hits3 = top1 = 0
        rr = 0.0
        for q, expected in tests:
            docs = [CHUNKS[i]["doc_id"] for i in fn(q)]
            first = next((pos for pos, dd in enumerate(docs) if dd in expected), None)
            if first is not None:
                rr += 1.0 / (first + 1)
                hits3 += first < 3
                top1 += first == 0
            elif verbose:
                print(f"  [{name}] aucun passage pertinent : {q}")
        n = len(tests)
        out[name] = {"recall@3": f"{hits3}/{n}", "top1": f"{top1}/{n}", "MRR": round(rr / n, 3)}
    return out


print(f"Index hybride construit : {len(CHUNKS)} passages, modèle {EMBEDDING_MODEL}")
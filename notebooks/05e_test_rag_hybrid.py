# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05e - Tests and comparison of retrievers (BM25 / Dense / Hybrid)
# MAGIC Requires `05d_rag_hybrid` in the same folder. Run the cells in order.

# COMMAND ----------

# MAGIC %pip install rank-bm25 sentence-transformers --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./05d_rag_hybrid

# COMMAND ----------

import json

# Sanity check: the right module was loaded (must print: True False)
print("CHUNKS" in globals(), "chunks" in globals())

# COMMAND ----------

# (query, set of relevant doc_ids)
TESTS = [
    ("à partir de quand un snapshot est obsolète", {"02_freshness_rules"}),
    ("les données sont trop vieilles", {"02_freshness_rules"}),
    ("quel écart est considéré normal", {"03_forecast_tolerance"}),
    ("la promo n'était pas synchronisée entre les deux systèmes", {"06_incident_2026_03_promo_sync"}),
    ("comment est calculée sales_7d", {"04_feature_definitions"}),
    ("le modèle a été déployé en version V3 d'un seul côté",
     {"05_model_versions_and_configs", "08_incident_2026_08_model_rollout"}),
    ("par quoi commencer pour investiguer un écart", {"09_runbook_investigation"}),
    ("pourquoi le batch est en retard par rapport au temps réel", {"02_freshness_rules"}),
    ("est-ce grave si les chiffres diffèrent un peu", {"03_forecast_tolerance"}),
    ("un des deux calculs utilise une autre réglage", {"05_model_versions_and_configs"}),
    ("que faire quand la prévision Batch est trop basse à cause d'un snapshot ancien",
     {"07_incident_2026_06_stale_snapshot", "09_runbook_investigation"}),
    ("de combien un horizon de 14 jours augmente la prévision", {"05_model_versions_and_configs"}),
    ("définition de DATA_MISMATCH", {"10_glossary_root_causes"}),
    ("durée typique d'exécution du pipeline temps réel", {"01_pipeline_overview"}),
]

results = evaluate_retrievers(TESTS)
for name, m in results.items():
    print(f"{name:8s}", m)

# COMMAND ----------

# Interface checks (same contract as the agent expects)
r = search_knowledge("un écart de 40 % avec un snapshot de 50 heures", 3)
print(json.dumps(r, indent=2, ensure_ascii=False)[:1800])
assert set(r) == {"query", "results"} and len(r["results"]) == 3
assert all({"doc_id", "section", "score", "text"} <= set(x) for x in r["results"])
assert "error" in search_knowledge("")
assert not any(len(c["body"]) < 40 for c in CHUNKS), "passage vide"
print("TESTS RAG HYBRIDE OK")

# COMMAND ----------

def show_misses(tests, top=3):
    rankers = {"BM25": rank_bm25, "Dense": rank_dense,
               "Hybride": lambda q: [i for i, _ in rrf([rank_bm25(q), rank_dense(q)])]}
    for name, fn in rankers.items():
        print(f"\n=== {name}: requêtes où le bon document n'est PAS dans le top {top} ===")
        found = False
        for q, expected in tests:
            order = fn(q)
            docs = [CHUNKS[i]["doc_id"] for i in order]
            first = next((p for p, d in enumerate(docs) if d in expected), None)
            if first is None or first >= top:
                found = True
                print(f"- {q}")
                print(f"    attendu : {sorted(expected)}")
                print(f"    rang du 1er bon passage : {None if first is None else first + 1}")
                print(f"    top {top} renvoyé : {[CHUNKS[i]['doc_id'] for i in order[:top]]}")
        if not found:
            print("(aucune)")

show_misses(TESTS)
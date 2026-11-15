# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 00 - Chargement du dataset synthétique (Forecast Debugging Agent)
# MAGIC 1. Importer le dossier `forecast_debug_dataset` dans un **Volume** Unity Catalog (Catalog > Volumes > Upload) ou dans le Workspace.
# MAGIC 2. Créer le schéma `forecast_debug` et le Volume `raw` (voir guide), puis exécuter toutes les cellules : BASE_PATH est calculé automatiquement.

# COMMAND ----------

import pandas as pd, glob, os

catalog = spark.sql("SELECT current_catalog()").first()[0]
schema = "forecast_debug"

# Le Volume "raw" doit contenir les dossiers data/ et rag_docs/
BASE_PATH = f"/Volumes/{catalog}/{schema}/raw"
print("BASE_PATH =", BASE_PATH)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# Tables structurées (6 tables Delta)
tables = ["forecast_predictions", "feature_snapshots", "model_configs",
          "pipeline_runs", "actual_sales", "known_issues"]

# Colonnes numériques à convertir (tout est lu en texte, puis converti)
numeric_cols = {
    "forecast_predictions": ["predicted_units"],
    "pipeline_runs": ["duration_seconds"],
    "actual_sales": ["actual_units"],
}

for t in tables:
    pdf = pd.read_csv(f"{BASE_PATH}/data/{t}.csv", dtype=str).fillna("")
    sdf = spark.createDataFrame(pdf)
    for c in numeric_cols.get(t, []):
        sdf = sdf.withColumn(c, sdf[c].cast("int"))
    (sdf.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.{t}"))
    print(f"Créée : {catalog}.{schema}.{t} ({len(pdf)} lignes)")

# COMMAND ----------

# Documents RAG -> table rag_documents
rows = []
for path in sorted(glob.glob(f"{BASE_PATH}/rag_docs/*.md")):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rows.append({"doc_id": os.path.basename(path).replace(".md", ""),
                 "title": text.splitlines()[0].lstrip("# ").strip(),
                 "content": text})

(spark.createDataFrame(pd.DataFrame(rows)).write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.rag_documents"))
print(f"Créée : {catalog}.{schema}.rag_documents ({len(rows)} documents)")

# COMMAND ----------

# Vérification : écart Batch vs Real-Time par cas
spark.sql(f"""
SELECT b.case_id, b.product_id,
       b.predicted_units AS batch, r.predicted_units AS realtime,
       r.predicted_units - b.predicted_units AS diff,
       ROUND(100.0 * (r.predicted_units - b.predicted_units) / b.predicted_units, 1) AS diff_pct
FROM {catalog}.{schema}.forecast_predictions b
JOIN {catalog}.{schema}.forecast_predictions r
  ON b.case_id = r.case_id AND b.system = 'BATCH' AND r.system = 'REALTIME'
ORDER BY b.case_id
""").display()
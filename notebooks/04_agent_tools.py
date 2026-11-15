# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 - Outils de l'agent (lecture seule)
# MAGIC 5 outils + un dispatcher sécurisé. À importer dans les autres notebooks avec `%run ./04_agent_tools`.
# MAGIC
# MAGIC **Principes**
# MAGIC - Lecture seule : aucune écriture, aucune requête SQL libre.
# MAGIC - Liste blanche de tables : `known_issues` (vérité terrain) et `actual_sales` sont **inaccessibles**.
# MAGIC - Les outils renvoient des **faits** (valeurs, âges en heures), **pas de verdict** (pas de "STALE", pas de "dans la tolérance").
# MAGIC   Les règles (seuil 24 h, tolérance 5 %) vivent dans les documents RAG, ce qui donne un vrai rôle au RAG.

# COMMAND ----------

import re, json
from pyspark.sql import SparkSession, functions as F

try:
    spark  # déjà défini dans un notebook Databricks
except NameError:
    spark = SparkSession.builder.getOrCreate()
CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "forecast_debug"
BASE = f"{CATALOG}.{SCHEMA}"

# Tables que les outils ont le droit de lire (known_issues et actual_sales sont volontairement absentes)
ALLOWED_TABLES = {
    "forecast_predictions", "feature_snapshots", "model_configs", "pipeline_runs",
    "cmp_forecast_gap", "cmp_inputs", "cmp_freshness",
}
CASE_ID_PATTERN = re.compile(r"^CASE_\d{3}$")


def _read_table(name):
    """Seul point d'accès aux données. Refuse toute table hors liste blanche."""
    if name not in ALLOWED_TABLES:
        raise PermissionError(f"Accès refusé à la table '{name}'")
    return spark.table(f"{BASE}.{name}")


def _check_case_id(case_id):
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.match(case_id):
        return {"error": f"case_id invalide : {case_id!r}. Format attendu : CASE_001"}
    return None


def _rows_for_case(table, case_id):
    df = _read_table(table).filter(F.col("case_id") == case_id)
    return [r.asDict() for r in df.collect()]


# COMMAND ----------

# Outil 1 : prévisions

def get_forecasts(case_id: str) -> dict:
    """Prévisions Batch et Real-Time d'un cas, avec l'écart absolu et relatif."""
    err = _check_case_id(case_id)
    if err:
        return err
    rows = _rows_for_case("forecast_predictions", case_id)
    if not rows:
        return {"error": f"Aucune prévision trouvée pour {case_id}"}
    by_system = {r["system"]: r for r in rows}
    if "BATCH" not in by_system or "REALTIME" not in by_system:
        return {"error": f"Prévision Batch ou Real-Time manquante pour {case_id}"}
    b, r = by_system["BATCH"], by_system["REALTIME"]
    diff = r["predicted_units"] - b["predicted_units"]
    return {
        "case_id": case_id,
        "product_id": b["product_id"],
        "forecast_date": b["forecast_date"],
        "batch_forecast": b["predicted_units"],
        "realtime_forecast": r["predicted_units"],
        "difference": diff,
        "difference_pct": round(100.0 * diff / b["predicted_units"], 2) if b["predicted_units"] else None,
    }


# COMMAND ----------

# Outil 2 : features

def compare_features(case_id: str) -> dict:
    """Compare les features (sales_7d, promotion, price, stock_level) entre Batch et Real-Time."""
    err = _check_case_id(case_id)
    if err:
        return err
    rows = [r for r in _rows_for_case("cmp_inputs", case_id) if r["finding_type"] == "FEATURE"]
    if not rows:
        return {"error": f"Aucune feature trouvée pour {case_id}"}
    features = sorted(
        [{"name": r["name"], "batch_value": r["batch_value"],
          "realtime_value": r["realtime_value"], "is_different": bool(r["is_difference"])} for r in rows],
        key=lambda x: x["name"])
    return {"case_id": case_id, "features": features,
            "differing_features": [f["name"] for f in features if f["is_different"]]}


# COMMAND ----------

# Outil 3 : configurations

def compare_configs(case_id: str) -> dict:
    """Compare les configurations (horizon, model_version, feature_set_version, preprocessing)."""
    err = _check_case_id(case_id)
    if err:
        return err
    rows = [r for r in _rows_for_case("cmp_inputs", case_id) if r["finding_type"] == "CONFIG"]
    if not rows:
        return {"error": f"Aucune configuration trouvée pour {case_id}"}
    configs = sorted(
        [{"name": r["name"], "batch_value": r["batch_value"],
          "realtime_value": r["realtime_value"], "is_different": bool(r["is_difference"])} for r in rows],
        key=lambda x: x["name"])
    return {"case_id": case_id, "configs": configs,
            "differing_configs": [c["name"] for c in configs if c["is_different"]]}


# COMMAND ----------

# Outil 4 : fraîcheur (faits uniquement, sans verdict)

def check_freshness(case_id: str) -> dict:
    """Âge des snapshots de données (en heures) au moment de l'exécution de chaque pipeline."""
    err = _check_case_id(case_id)
    if err:
        return err
    rows = _rows_for_case("cmp_freshness", case_id)
    if not rows:
        return {"error": f"Aucune donnée de fraîcheur pour {case_id}"}
    r = rows[0]
    return {
        "case_id": case_id,
        "batch_snapshot": str(r["batch_snapshot"]),
        "realtime_snapshot": str(r["realtime_snapshot"]),
        "batch_age_hours": round(r["batch_age_hours"], 2),
        "realtime_age_hours": round(r["realtime_age_hours"], 2),
        "snapshot_gap_hours": r["snapshot_gap_hours"],
    }


# COMMAND ----------

# Outil 5 : logs de pipeline

def get_pipeline_logs(case_id: str) -> dict:
    """Journaux d'exécution des deux pipelines : statut, warnings, versions, durée."""
    err = _check_case_id(case_id)
    if err:
        return err
    rows = _rows_for_case("pipeline_runs", case_id)
    if not rows:
        return {"error": f"Aucun log trouvé pour {case_id}"}
    out = {"case_id": case_id}
    for r in rows:
        out[r["system"].lower()] = {
            "run_id": r["run_id"],
            "run_timestamp": str(r["run_timestamp"]),
            "data_snapshot_timestamp": str(r["data_snapshot_timestamp"]),
            "status": r["status"],
            "warning_message": r["warning_message"] or None,
            "code_version": r["code_version"],
            "data_version": r["data_version"],
            "duration_seconds": r["duration_seconds"],
        }
    return out


# COMMAND ----------

# Registre, schémas (format compatible tool calling) et dispatcher sécurisé

TOOLS = {
    "get_forecasts": get_forecasts,
    "compare_features": compare_features,
    "compare_configs": compare_configs,
    "check_freshness": check_freshness,
    "get_pipeline_logs": get_pipeline_logs,
}

def _schema(name, description):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object",
                       "properties": {"case_id": {"type": "string", "description": "Identifiant du cas, ex. CASE_001"}},
                       "required": ["case_id"]}}}

TOOL_SCHEMAS = [
    _schema("get_forecasts", "Retourne les prévisions Batch et Real-Time d'un cas et l'écart (unités et %)."),
    _schema("compare_features", "Compare les features d'entrée (sales_7d, promotion, price, stock_level) entre Batch et Real-Time."),
    _schema("compare_configs", "Compare les configurations des modèles (horizon, model_version, feature_set_version, preprocessing)."),
    _schema("check_freshness", "Retourne l'âge en heures des snapshots de données de chaque pipeline. Ne donne aucun verdict : appliquer les règles de fraîcheur de la documentation."),
    _schema("get_pipeline_logs", "Retourne les journaux d'exécution : statut, warnings, versions de code et de données, durée."),
]


def call_tool(name, arguments):
    """Exécute un outil demandé par le LLM. Retourne toujours une chaîne JSON."""
    if name not in TOOLS:
        return json.dumps({"error": f"Outil inconnu : {name}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        return json.dumps(TOOLS[name](**args), default=str, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"error": f"Arguments invalides : {e}"})
    except Exception as e:
        return json.dumps({"error": f"Erreur outil : {type(e).__name__}: {e}"})

print("Outils chargés :", list(TOOLS), "| Schéma :", BASE)
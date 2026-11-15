# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 - Baseline sans LLM : diagnostic par règles
# MAGIC Lit **uniquement** les tables `cmp_*` (jamais `known_issues`) et applique les règles du RAG.
# MAGIC `known_issues` n'est lue que dans la dernière section, réservée à l'évaluation.

# COMMAND ----------

catalog = spark.sql("SELECT current_catalog()").first()[0]
base = f"{catalog}.forecast_debug"

# Priorité entre causes quand plusieurs différences existent (la 1re est la cause principale)
PRIORITY = ["STALE_DATA", "FEATURE_MISMATCH", "MODEL_VERSION_MISMATCH", "CONFIG_MISMATCH", "DATA_MISMATCH"]

def classify_difference(finding_type, name):
    """Associe une différence d'entrée à une catégorie de cause."""
    if finding_type == "CONFIG":
        return "MODEL_VERSION_MISMATCH" if name == "model_version" else "CONFIG_MISMATCH"
    return "DATA_MISMATCH" if name == "sales_7d" else "FEATURE_MISMATCH"

def investigate_case(case_id):
    """Diagnostic par règles pour un cas. Retourne un dictionnaire avec preuves."""
    summary = spark.table(f"{base}.cmp_case_summary").filter(f"case_id = '{case_id}'").first()
    fresh = spark.table(f"{base}.cmp_freshness").filter(f"case_id = '{case_id}'").first()
    diffs = [r.asDict() for r in spark.table(f"{base}.cmp_inputs")
             .filter(f"case_id = '{case_id}' AND is_difference = true").collect()]

    evidence = [f"Écart: {summary['difference']} unités ({summary['difference_pct']} %)"]
    causes = {}

    # Règle 1 : snapshot obsolète (> 24 h). Les différences de features deviennent des symptômes.
    stale = fresh["freshness_status"] == "STALE"
    if stale:
        causes["STALE_DATA"] = True
        evidence.append(f"Âge du snapshot Batch: {fresh['batch_age_hours']:.0f} h (seuil 24 h)"
                        if fresh["batch_age_hours"] > 24 else
                        f"Âge du snapshot Real-Time: {fresh['realtime_age_hours']:.0f} h (seuil 24 h)")
        if fresh["batch_warning"]:
            evidence.append(f"Warning pipeline: {fresh['batch_warning']}")

    # Règle 2 : différences d'entrée
    for d in diffs:
        cause = classify_difference(d["finding_type"], d["name"])
        if stale and cause == "DATA_MISMATCH":
            evidence.append(f"{d['name']}: {d['batch_value']} vs {d['realtime_value']} (symptôme de l'obsolescence)")
            continue
        causes[cause] = True
        evidence.append(f"{d['name']}: Batch={d['batch_value']} vs Real-Time={d['realtime_value']}")

    ordered = [c for c in PRIORITY if c in causes]

    # Règle 3 : aucune cause et écart dans la tolérance -> pas de problème
    if not ordered and summary["within_tolerance"]:
        return {"case_id": case_id, "primary_cause": "NO_ISSUE", "secondary_causes": [],
                "evidence": evidence + [f"Écart sous la tolérance de 5 %, aucune différence d'entrée"],
                "confidence": "high"}

    # Règle 4 : écart important mais aucune cause trouvée
    if not ordered:
        return {"case_id": case_id, "primary_cause": "UNKNOWN", "secondary_causes": [],
                "evidence": evidence + ["Écart hors tolérance sans différence détectée"], "confidence": "low"}

    return {"case_id": case_id, "primary_cause": ordered[0], "secondary_causes": ordered[1:],
            "evidence": evidence, "confidence": "high" if len(ordered) == 1 else "medium"}

# COMMAND ----------

# Test sur un cas
import json
print(json.dumps(investigate_case("CASE_009"), indent=2, ensure_ascii=False))

# COMMAND ----------

# Lancer sur les 12 cas
case_ids = [r["case_id"] for r in spark.table(f"{base}.cmp_case_summary").orderBy("case_id").collect()]
results = [investigate_case(c) for c in case_ids]
for r in results:
    print(r["case_id"], "->", r["primary_cause"], r["secondary_causes"], f"({r['confidence']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ÉVALUATION (réservé : lecture de la vérité terrain)

# COMMAND ----------

truth = {r["case_id"]: r.asDict() for r in spark.table(f"{base}.known_issues").collect()}

correct = 0
for r in results:
    t = truth[r["case_id"]]
    ok = r["primary_cause"] == t["expected_root_cause"]
    correct += ok
    print(("OK   " if ok else "ERREUR"), r["case_id"], "prédit:", r["primary_cause"], "| attendu:", t["expected_root_cause"])

print(f"\nAccuracy baseline règles: {correct}/{len(results)} = {100*correct/len(results):.1f} %")
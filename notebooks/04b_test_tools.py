# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04b - Tests des outils de l'agent

# COMMAND ----------

# MAGIC %run ./04_agent_tools

# COMMAND ----------

import json

# 1. Aperçu des 5 outils sur CASE_004 (données obsolètes)
for name in TOOLS:
    print(f"--- {name} ---")
    print(json.dumps(TOOLS[name]("CASE_004"), indent=2, ensure_ascii=False))

# COMMAND ----------

# 2. Tests automatiques
r = get_forecasts("CASE_002")
assert r["difference"] == 350 and r["difference_pct"] == 35.0, r

assert compare_features("CASE_002")["differing_features"] == ["promotion"]
assert compare_features("CASE_007")["differing_features"] == []
assert compare_configs("CASE_003")["differing_configs"] == ["forecast_horizon_days"]
assert compare_configs("CASE_008")["differing_configs"] == ["model_version"]

f = check_freshness("CASE_004")
assert abs(f["batch_age_hours"] - 50.0) < 0.01, f
assert "freshness_status" not in f, "Les outils ne doivent pas donner de verdict"

logs = get_pipeline_logs("CASE_004")
assert "sales_daily" in logs["batch"]["warning_message"], logs
assert logs["realtime"]["warning_message"] is None

# Entrées invalides : erreur propre, pas d'exception
assert "error" in get_forecasts("CASE_999")
assert "error" in get_forecasts("'; DROP TABLE x; --")
assert "error" in get_forecasts(None)

# Sécurité : la vérité terrain est inaccessible
for forbidden in ["known_issues", "actual_sales", "rag_documents"]:
    try:
        _read_table(forbidden)
        raise AssertionError(f"{forbidden} ne devrait pas être lisible")
    except PermissionError:
        pass

# Dispatcher
assert "error" in json.loads(call_tool("drop_table", {}))
assert "error" in json.loads(call_tool("get_forecasts", {"mauvais": 1}))
assert json.loads(call_tool("get_forecasts", '{"case_id": "CASE_001"}'))["difference"] == 200

# Schémas
assert len(TOOL_SCHEMAS) == 5 and all(s["function"]["name"] in TOOLS for s in TOOL_SCHEMAS)

# 3. Les 5 outils sur les 12 cas, sans erreur
for i in range(1, 13):
    cid = f"CASE_{i:03d}"
    for name, fn in TOOLS.items():
        res = fn(cid)
        assert "error" not in res, (name, cid, res)
        json.dumps(res, default=str)

print("TOUS LES TESTS OK")
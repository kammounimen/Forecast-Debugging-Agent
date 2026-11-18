# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 07 - Single agent (configuration A)
# MAGIC One LLM, 5 read-only tools, optional `search_knowledge` (RAG). Needs in the same folder: `04_agent_tools`, `05d_rag_hybrid`.
# MAGIC
# MAGIC - `USE_RAG = True`  : agent + RAG (rules and definitions come from the documents)
# MAGIC - `USE_RAG = False` : agent alone (no access to the 24 h / 5 % rules)
# MAGIC
# MAGIC The agent never reads `known_issues`. Only the last cell does, to score the answers.

# COMMAND ----------

# MAGIC %pip install rank-bm25 sentence-transformers openai databricks-sdk --upgrade --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./04_agent_tools

# COMMAND ----------

# MAGIC %run ./05d_rag_hybrid

# COMMAND ----------

# Config: change USE_RAG and rerun from this cell onward to compare configurations
import json, re, time

MODEL = "databricks-gpt-oss-120b"
USE_RAG = True
MAX_STEPS = 10
CASES = [f"CASE_{i:03d}" for i in range(1, 13)]
CAUSES = ["DATA_MISMATCH", "FEATURE_MISMATCH", "CONFIG_MISMATCH",
          "MODEL_VERSION_MISMATCH", "STALE_DATA", "NO_ISSUE"]

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

# COMMAND ----------

SYSTEM_TEMPLATE = """Tu es un agent d'investigation des écarts entre les prévisions Batch et Real-Time.
Pour le cas demandé, enquête avec les outils, puis conclus.

Méthode :
1. get_forecasts : mesure l'écart.
2. check_freshness et get_pipeline_logs : âges des snapshots, statut et warnings du pipeline.
3. compare_features et compare_configs : ce qui diffère entre les deux pipelines.
{RAG}

Règles :
- Ne devine pas. Chaque élément de preuve doit être une valeur retournée par un outil.
- Un symptôme n'est pas une cause racine : si une différence est la conséquence d'une autre, la cause racine est l'autre.
- Si aucune cause n'est établie par les faits, la réponse est NO_ISSUE.
- Une seule cause principale ; les autres causes éventuelles vont dans secondary_causes.

Causes possibles : DATA_MISMATCH, FEATURE_MISMATCH, CONFIG_MISMATCH, MODEL_VERSION_MISMATCH, STALE_DATA, NO_ISSUE.

Réponse finale : UNIQUEMENT un objet JSON, sans texte autour, de la forme :
{"primary_cause": "...", "secondary_causes": [], "evidence": ["..."], "confidence": "high|medium|low"}"""

RAG_ON = ("4. search_knowledge : consulte la documentation interne (seuils, définitions, incidents passés) "
          "pour interpréter les faits. Cherche avec des termes concrets : nom de la feature, valeur observée, durée.")
RAG_OFF = "Tu n'as pas accès à la documentation interne : n'invente aucun seuil."


def make_system():
    return SYSTEM_TEMPLATE.replace("{RAG}", RAG_ON if USE_RAG else RAG_OFF)


def dispatch(name, args):
    if name == "search_knowledge":
        if not USE_RAG:
            return {"error": "outil non disponible"}
        return search_knowledge(args.get("query"), args.get("k", 3))
    return call_tool(name, args)


def _text(msg):
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else (getattr(p, "text", "") or "") for p in c)
    return c or ""


def _chat(messages, tools):
    for attempt in range(4):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools,
                tool_choice="auto", temperature=0, max_tokens=3000)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def _parse_json(text):
    try:
        s, e = text.index("{"), text.rindex("}")
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def run_agent(case_id, verbose=False):
    tools = list(TOOL_SCHEMAS) + ([SEARCH_KNOWLEDGE_SCHEMA] if USE_RAG else [])
    messages = [{"role": "system", "content": make_system()},
                {"role": "user", "content": f"Investigue {case_id}."}]
    called = []
    for step in range(1, MAX_STEPS + 1):
        msg = _chat(messages, tools).choices[0].message
        if not msg.tool_calls:
            raw = _text(msg)
            return {"case_id": case_id, "report": _parse_json(raw), "raw": raw,
                    "tools": called, "steps": step, "error": None}
        messages.append({"role": "assistant", "content": _text(msg),
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if verbose:
                print(f"  [{case_id} step {step}] {tc.function.name}({args})")
            called.append(tc.function.name)
            result = dispatch(tc.function.name, args)
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return {"case_id": case_id, "report": None, "raw": "", "tools": called,
            "steps": MAX_STEPS, "error": "max_steps"}

# COMMAND ----------

# Try one case first (CASE_009: stale snapshot, sales_7d is a symptom)
r = run_agent("CASE_009", verbose=True)
print(json.dumps(r["report"], indent=2, ensure_ascii=False))
print("tools:", r["tools"])

# COMMAND ----------

# Run the 12 cases
results = []
for c in CASES:
    r = run_agent(c)
    results.append(r)
    rep = r["report"] or {}
    print(f"{c} -> {rep.get('primary_cause')} {rep.get('secondary_causes', [])} "
          f"({rep.get('confidence')}) | {len(r['tools'])} tool calls"
          + (f" | ERROR {r['error']}" if r["error"] else "")
          + ("" if r["report"] else " | JSON NOT PARSED"))

# COMMAND ----------

# EVALUATION (reserved cell: the only place that reads the ground truth)
from datetime import datetime

cat = spark.sql("SELECT current_catalog()").first()[0]
truth_df = spark.table(f"{cat}.forecast_debug.known_issues")
print("known_issues columns:", truth_df.columns)

names = {c.lower(): c for c in truth_df.columns}
cause_col = next((names[n] for n in ("root_cause", "expected_cause", "true_cause", "cause", "issue_type")
                  if n in names), None) or next((c for c in truth_df.columns
                                                 if "cause" in c.lower() and c.lower() != "case_id"), None)
if cause_col is None:
    raise RuntimeError(f"Cannot find the cause column in {truth_df.columns}. Set cause_col by hand.")
truth = {row["case_id"]: row[cause_col] for row in truth_df.collect()}

ok = 0
rows = []
config = f"single_{'rag' if USE_RAG else 'norag'}_{MODEL}"
for r in results:
    rep = r["report"] or {}
    pred = rep.get("primary_cause")
    pred = pred if pred in CAUSES else "INVALID"
    good = pred == truth.get(r["case_id"])
    ok += good
    print(f"{'OK ' if good else 'ERR'} {r['case_id']} predicted: {pred:24s} | expected: {truth.get(r['case_id'])}")
    rows.append((config, r["case_id"], pred, json.dumps(rep.get("secondary_causes", [])),
                 str(rep.get("confidence")), len(r["tools"]), ",".join(r["tools"]),
                 bool(good), r["raw"][:2000], datetime.utcnow().isoformat()))

n = len(results)
print(f"\nAccuracy {config}: {ok}/{n} = {100 * ok / n:.1f} %")
print(f"Average tool calls per case: {sum(len(r['tools']) for r in results) / n:.1f}")

try:
    spark.createDataFrame(rows, "config string, case_id string, predicted string, secondary string, "
                                "confidence string, n_tools int, tools string, correct boolean, "
                                "raw string, ts string") \
         .write.mode("append").saveAsTable(f"{cat}.forecast_debug.agent_runs")
    print(f"Saved to {cat}.forecast_debug.agent_runs")
except Exception as e:
    print("Could not save the run:", type(e).__name__, str(e)[:200])
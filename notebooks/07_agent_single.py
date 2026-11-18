# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC **Ordre important : `05d_rag_hybrid` d'abord, puis `04_agent_tools` (chargé en dernier). Chaque `%run` doit être seul, sans commentaire avant, dans sa cellule.**
# MAGIC
# MAGIC # 07 - Agent unique (configuration A), version corrigée
# MAGIC Un LLM, 5 outils de données en lecture seule, et `search_knowledge` (RAG) en option.
# MAGIC À mettre dans le même dossier que `04_agent_tools` et `05d_rag_hybrid`.
# MAGIC
# MAGIC - `USE_RAG = True`  : agent + RAG
# MAGIC - `USE_RAG = False` : agent seul (pas d'accès aux règles 24 h / 5 %)
# MAGIC - `N_RUNS` : nombre d'exécutions complètes des 12 cas (un LLM n'est pas parfaitement reproductible)
# MAGIC
# MAGIC **Ce qui change par rapport à la version précédente**
# MAGIC 1. Réponse finale vide ou illisible : une seule relance sans outils, **signalée** (`repaired`), la réponse brute d'origine est conservée.
# MAGIC 2. Un cas qui plante n'arrête plus la boucle.
# MAGIC 3. Plusieurs exécutions, avec stabilité par cas.
# MAGIC 4. Métriques : exactitude de la cause principale, précision/rappel sur l'ensemble des causes, appels LLM, temps, ancrage des preuves.
# MAGIC 5. Résultats sauvegardés dans `agent_runs_v2` (nouvelle table, l'ancienne `agent_runs` n'est pas touchée).
# MAGIC
# MAGIC Le prompt n'a **pas** été modifié (`PROMPT_VERSION = "v1"`). Si vous le changez, changez aussi ce nom.
# MAGIC L'agent ne lit jamais `known_issues` : seule la cellule d'évaluation le fait.

# COMMAND ----------

# MAGIC %pip install rank-bm25 sentence-transformers openai databricks-sdk --upgrade --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./05d_rag_hybrid

# COMMAND ----------

# MAGIC %run ./04_agent_tools

# COMMAND ----------

import json
needed = ["call_tool", "TOOL_SCHEMAS", "search_knowledge", "SEARCH_KNOWLEDGE_SCHEMA"]
for n in needed:
    print(f"{n:25s}", n in globals())
assert all(n in globals() for n in needed), "Une définition manque : vérifiez l'ordre des %run."
for t in TOOL_SCHEMAS:
    name = t["function"]["name"]
    r = json.loads(call_tool(name, {"case_id": "CASE_004"}))
    print(f"  {name:20s}", "ERREUR : " + str(r)[:120] if "error" in r else "OK")
print("search_knowledge     ", "OK" if "results" in search_knowledge("snapshot obsolète 24 h") else "ERREUR")

# COMMAND ----------

# Configuration : modifiez USE_RAG / N_RUNS, puis relancez à partir de cette cellule
import json, re, time, uuid, warnings
from datetime import datetime, timezone

MODEL = "databricks-gpt-oss-120b"
USE_RAG = True
N_RUNS = 3            # mettez 3 pour la mesure officielle
MAX_STEPS = 10
MAX_TOKENS = 4000     # un modèle à raisonnement consomme des jetons avant de répondre
PROMPT_VERSION = "v1"
CASES = [f"CASE_{i:03d}" for i in range(1, 13)]
CAUSES = ["DATA_MISMATCH", "FEATURE_MISMATCH", "CONFIG_MISMATCH",
          "MODEL_VERSION_MISMATCH", "STALE_DATA", "NO_ISSUE"]
CONFIDENCES = {"high", "medium", "low"}

from databricks.sdk import WorkspaceClient
warnings.filterwarnings("ignore", category=DeprecationWarning)
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

REPAIR_MESSAGE = ("Ta réponse précédente n'est pas un objet JSON exploitable. À partir des résultats d'outils "
                  "ci-dessus, réponds maintenant UNIQUEMENT par l'objet JSON demandé, sans autre texte.")


def make_system():
    return SYSTEM_TEMPLATE.replace("{RAG}", RAG_ON if USE_RAG else RAG_OFF)


def dispatch(name, args):
    if name == "search_knowledge":
        if not USE_RAG:
            return json.dumps({"error": "outil non disponible"})
        return json.dumps(search_knowledge(args.get("query"), args.get("k", 3)), ensure_ascii=False, default=str)
    return call_tool(name, args)          # call_tool renvoie déjà une chaîne JSON


def _text(msg):
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else (getattr(p, "text", "") or "") for p in c)
    return c or ""


def _chat(messages, tools=None):
    kwargs = dict(model=MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS)
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def _parse_report(text):
    """Extrait et normalise l'objet JSON. Retourne None s'il est absent ou invalide."""
    try:
        s, e = text.index("{"), text.rindex("}")
        rep = json.loads(text[s:e + 1])
    except Exception:
        return None
    if not isinstance(rep, dict) or rep.get("primary_cause") not in CAUSES:
        return None
    sec = rep.get("secondary_causes") or []
    sec = [c for c in sec if c in CAUSES and c != rep["primary_cause"]] if isinstance(sec, list) else []
    ev = rep.get("evidence") or []
    return {"primary_cause": rep["primary_cause"],
            "secondary_causes": list(dict.fromkeys(sec)),
            "evidence": [str(x) for x in ev] if isinstance(ev, list) else [str(ev)],
            "confidence": rep.get("confidence") if rep.get("confidence") in CONFIDENCES else None}


def run_agent(case_id, verbose=False):
    t0 = time.time()
    tools = list(TOOL_SCHEMAS) + ([SEARCH_KNOWLEDGE_SCHEMA] if USE_RAG else [])
    messages = [{"role": "system", "content": make_system()},
                {"role": "user", "content": f"Investigue {case_id}."}]
    called, outputs, n_llm = [], [], 0
    out = {"case_id": case_id, "report": None, "raw": "", "first_raw": "", "repaired": False,
           "tools": called, "tool_outputs": outputs, "steps": 0, "n_llm": 0,
           "finish_reason": None, "error": None, "seconds": 0.0}

    for step in range(1, MAX_STEPS + 1):
        resp = _chat(messages, tools)
        n_llm += 1
        choice = resp.choices[0]
        msg = choice.message
        out["steps"], out["finish_reason"] = step, choice.finish_reason

        if not msg.tool_calls:                               # réponse finale
            raw = _text(msg)
            out["first_raw"] = raw
            report = _parse_report(raw)
            if report is None:                               # une seule relance, sans outils
                out["repaired"] = True
                if raw.strip():
                    messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": REPAIR_MESSAGE})
                resp2 = _chat(messages, None)
                n_llm += 1
                raw = _text(resp2.choices[0].message)
                out["finish_reason"] = resp2.choices[0].finish_reason
                report = _parse_report(raw)
            out.update(report=report, raw=raw)
            break

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
            content = dispatch(tc.function.name, args)
            outputs.append((tc.function.name, content))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    else:
        out["error"] = "max_steps"

    out["n_llm"] = n_llm
    out["seconds"] = round(time.time() - t0, 1)
    return out

# COMMAND ----------

# Test sur un cas (CASE_009 : snapshot obsolète, sales_7d n'est qu'un symptôme)
r = run_agent("CASE_009", verbose=True)
print(json.dumps(r["report"], indent=2, ensure_ascii=False))
print("outils :", r["tools"], "| appels LLM :", r["n_llm"], "| relancé :", r["repaired"], f"| {r['seconds']} s")

# COMMAND ----------

# Exécution des cas : N_RUNS passes complètes. Un cas qui plante est noté, il n'arrête pas la boucle.
BATCH_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
results = []
for run_idx in range(1, N_RUNS + 1):
    print(f"===== passe {run_idx}/{N_RUNS} =====")
    for c in CASES:
        try:
            r = run_agent(c)
        except Exception as e:
            r = {"case_id": c, "report": None, "raw": "", "first_raw": "", "repaired": False,
                 "tools": [], "tool_outputs": [], "steps": 0, "n_llm": 0, "finish_reason": None,
                 "error": f"{type(e).__name__}: {str(e)[:150]}", "seconds": 0.0}
        r["run"] = run_idx
        results.append(r)
        rep = r["report"] or {}
        flags = ((" | RELANCÉ" if r["repaired"] else "")
                 + (f" | ERREUR {r['error']}" if r["error"] else "")
                 + ("" if r["report"] else " | JSON NON LU"))
        print(f"{c} -> {rep.get('primary_cause')} {rep.get('secondary_causes', [])} "
              f"({rep.get('confidence')}) | {len(r['tools'])} outils, {r['n_llm']} appels LLM, {r['seconds']} s{flags}")

# COMMAND ----------

# Diagnostic des réponses illisibles ou relancées : à regarder avant d'interpréter les scores
bad = [r for r in results if r["repaired"] or r["report"] is None]
if not bad:
    print("Aucune réponse relancée ni illisible.")
for r in bad:
    print(f"\n--- {r['case_id']} (passe {r['run']}) | relancé : {r['repaired']} | "
          f"finish_reason : {r['finish_reason']} | erreur : {r['error']}")
    print("1re réponse brute :", repr(r["first_raw"])[:600])
    print("réponse finale    :", repr(r["raw"])[:600])

# COMMAND ----------

# ÉVALUATION (cellule réservée : la seule qui lit la vérité terrain)
import statistics
from collections import defaultdict

cat = spark.sql("SELECT current_catalog()").first()[0]
truth_df = spark.table(f"{cat}.forecast_debug.known_issues")
print("colonnes de known_issues :", truth_df.columns)
print("valeurs de secondary_cause :", sorted({str(r['secondary_cause']) for r in truth_df.collect()}))

truth, truth_sec = {}, {}
NONE_LIKE = {None, "", "NONE", "None", "none", "null", "NULL", "NO_ISSUE"}
for row in truth_df.collect():
    truth[row["case_id"]] = row["expected_root_cause"]
    truth_sec[row["case_id"]] = row["secondary_cause"]


def expected_set(case_id):
    return {c for c in (truth[case_id], truth_sec.get(case_id)) if c not in NONE_LIKE} or {"NO_ISSUE"}


def predicted_set(rep):
    if not rep:
        return set()
    return {rep["primary_cause"], *rep.get("secondary_causes", [])}


# Ancrage approximatif des preuves : les nombres cités sont-ils présents dans les sorties d'outils ?
def _nums(s):
    s = s.replace("\u202f", "").replace("\u00a0", "")
    out = set()
    for x in re.findall(r"\d+(?:[.,]\d+)?", s):
        try:
            out.add(round(float(x.replace(",", ".")), 2))
        except ValueError:
            pass
    return out


def grounding(r):
    if not r["report"]:
        return None
    pool = set()
    for _, content in r["tool_outputs"]:
        pool |= _nums(content)
    cited = set()
    for e in r["report"]["evidence"]:
        cited |= _nums(e)
    return (len(cited & pool), len(cited)) if cited else (0, 0)


config = f"single_{'rag' if USE_RAG else 'norag'}_{PROMPT_VERSION}_{MODEL}"
by_run, rows = defaultdict(list), []
tp = fp = fn = 0
gr_ok = gr_all = 0

for r in results:
    cid = r["case_id"]
    rep = r["report"]
    pred = rep["primary_cause"] if rep else "INVALID"
    good = pred == truth[cid]
    by_run[r["run"]].append(good)
    exp, prd = expected_set(cid), predicted_set(rep)
    tp += len(exp & prd); fp += len(prd - exp); fn += len(exp - prd)
    g = grounding(r)
    if g:
        gr_ok += g[0]; gr_all += g[1]
    print(f"{'OK ' if good else 'ERR'} passe {r['run']} {cid} prédit : {pred:24s} | attendu : {truth[cid]}"
          + ("" if exp == prd else f" | ensemble attendu {sorted(exp)} vs prédit {sorted(prd)}"))
    rows.append((BATCH_ID, config, r["run"], cid, pred, json.dumps((rep or {}).get("secondary_causes", [])),
                 str((rep or {}).get("confidence")), len(r["tools"]), ",".join(r["tools"]), r["n_llm"],
                 float(r["seconds"]), bool(r["repaired"]), bool(good),
                 (r["raw"] or "")[:2000], (r["first_raw"] or "")[:2000],
                 json.dumps(r["tool_outputs"], ensure_ascii=False)[:12000],
                 datetime.now(timezone.utc).isoformat()))

accs = [sum(v) / len(v) for v in by_run.values()]
print("\n=== RÉSUMÉ", config, "===")
print("Exactitude cause principale par passe :", [f"{sum(v)}/{len(v)}" for v in by_run.values()])
print(f"Moyenne : {100 * statistics.mean(accs):.1f} %"
      + (f" | min {100 * min(accs):.1f} % | max {100 * max(accs):.1f} %" if len(accs) > 1 else ""))
if tp + fp and tp + fn:
    print(f"Ensemble de causes (toutes passes) : précision {tp / (tp + fp):.2f} | rappel {tp / (tp + fn):.2f}")
n = len(results)
print(f"Appels d'outils moyens : {sum(len(r['tools']) for r in results) / n:.1f} | "
      f"appels LLM moyens : {sum(r['n_llm'] for r in results) / n:.1f} | "
      f"temps moyen : {sum(r['seconds'] for r in results) / n:.1f} s")
print(f"Réponses relancées : {sum(r['repaired'] for r in results)}/{n} | "
      f"illisibles : {sum(r['report'] is None for r in results)}/{n}")
if gr_all:
    print(f"Ancrage approximatif des preuves : {gr_ok}/{gr_all} nombres cités retrouvés dans les sorties d'outils")

if N_RUNS > 1:
    print("\nStabilité par cas (causes principales prédites sur les passes) :")
    for cid in CASES:
        preds = [(r["report"] or {}).get("primary_cause", "INVALID") for r in results if r["case_id"] == cid]
        if len(set(preds)) > 1:
            print(f"  {cid} INSTABLE : {preds}")

try:
    schema = ("batch_id string, config string, run int, case_id string, predicted string, secondary string, "
              "confidence string, n_tools int, tools string, n_llm int, seconds double, repaired boolean, "
              "correct boolean, raw string, first_raw string, tool_outputs string, ts string")
    spark.createDataFrame(rows, schema).write.mode("append").saveAsTable(f"{cat}.forecast_debug.agent_runs_v2")
    print(f"\nSauvegardé dans {cat}.forecast_debug.agent_runs_v2 (batch {BATCH_ID})")
except Exception as e:
    print("\nSauvegarde impossible :", type(e).__name__, str(e)[:200])
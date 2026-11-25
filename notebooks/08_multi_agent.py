# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC **Ordre important : `05d_rag_hybrid` d'abord, puis `04_agent_tools` (chargé en dernier). Chaque `%run` doit être seul, sans commentaire avant, dans sa cellule.**
# MAGIC
# MAGIC # 08 - Système à 3 agents (LangGraph)
# MAGIC
# MAGIC ```
# MAGIC Question -> [Agent 1 Investigateur]  \
# MAGIC             (5 outils de données)     >-- en parallèle --> [Agent 3 Analyste + Vérificateur] -> Rapport final
# MAGIC          -> [Agent 2 Connaissance]  /
# MAGIC             (RAG hybride)
# MAGIC ```
# MAGIC
# MAGIC - **Agent 1** collecte les faits avec les 5 outils (lecture seule). Il ne conclut pas.
# MAGIC - **Agent 2** retrouve dans la documentation les règles, définitions, incidents passés et runbooks utiles.
# MAGIC - **Agent 3** croise les deux, choisit la cause racine, cite une source pour chaque preuve, puis le **code** vérifie les citations (sources valides, nombres retrouvés). Une seule révision est demandée en cas de problème, et le résultat de la vérification est conservé dans le rapport.
# MAGIC
# MAGIC Rapport final : `primary_cause`, `secondary_causes`, `evidence` (affirmation + source), `confidence`, `recommendation`, `verification`.
# MAGIC `known_issues` n'est lu que par la dernière cellule, pour mesurer.

# COMMAND ----------

# MAGIC %pip install langgraph rank-bm25 sentence-transformers openai databricks-sdk --upgrade --quiet

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

# Configuration
import json, re, time, uuid, warnings
from datetime import datetime, timezone

MODEL = "databricks-gpt-oss-120b"     # même modèle que l'agent unique
USE_RAG = True                         # False : l'Agent 2 est désactivé
PARALLEL = True                        # False : Agents 1 puis 2 en séquence (si le parallèle pose problème)
N_RUNS = 1                             # mettez 3 pour avoir 3 passes (environ 12 minutes)
MAX_STEPS = 10
MAX_TOKENS = 4000
PROMPT_VERSION = "ma_v2"
CASES = [f"CASE_{i:03d}" for i in range(1, 13)]
CAUSES = ["DATA_MISMATCH", "FEATURE_MISMATCH", "CONFIG_MISMATCH",
          "MODEL_VERSION_MISMATCH", "STALE_DATA", "NO_ISSUE"]
CONFIDENCES = {"high", "medium", "low"}

from databricks.sdk import WorkspaceClient
warnings.filterwarnings("ignore", category=DeprecationWarning)
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

# COMMAND ----------

# Outils communs : appel LLM, boucle d'outils, extraction JSON
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


def _parse_obj(text):
    try:
        s, e = text.index("{"), text.rindex("}")
        obj = json.loads(text[s:e + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


REPAIR_MESSAGE = ("Ta réponse précédente n'est pas un objet JSON exploitable. À partir des résultats ci-dessus, "
                  "réponds maintenant UNIQUEMENT par l'objet JSON demandé, sans autre texte.")


def run_llm_agent(system, user, tools, dispatch_fn, validate=None):
    """Boucle d'outils générique. Une seule relance de format si la réponse finale est vide ou invalide."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    called, outputs, n_llm = [], [], 0
    res = {"obj": None, "raw": "", "repaired": False, "tools": called, "tool_outputs": outputs,
           "n_llm": 0, "error": None}
    for step in range(1, MAX_STEPS + 1):
        resp = _chat(messages, tools)
        n_llm += 1
        msg = resp.choices[0].message
        if not msg.tool_calls:
            raw = _text(msg)
            obj = _parse_obj(raw)
            if obj is None or (validate and not validate(obj)):
                res["repaired"] = True
                if raw.strip():
                    messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": REPAIR_MESSAGE})
                raw = _text(_chat(messages, None).choices[0].message)
                n_llm += 1
                obj = _parse_obj(raw)
                if obj is not None and validate and not validate(obj):
                    obj = None
            res.update(obj=obj, raw=raw)
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
            called.append(tc.function.name)
            content = dispatch_fn(tc.function.name, args)
            outputs.append((tc.function.name, content))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    else:
        res["error"] = "max_steps"
    res["n_llm"] = n_llm
    return res

# COMMAND ----------

# Les prompts des 3 agents
INVESTIGATOR_SYSTEM = """Tu es l'Agent 1 (Investigateur). Tu collectes des FAITS sur l'écart entre les prévisions Batch et Real-Time d'un cas.
Utilise les 5 outils : get_forecasts, check_freshness, get_pipeline_logs, compare_features, compare_configs. Appelle-les tous.
Tu ne conclus PAS sur la cause racine et tu n'appliques aucun seuil : un autre agent s'en charge.

Réponse finale : UNIQUEMENT un objet JSON de la forme :
{"facts": [{"fact": "phrase courte avec les valeurs exactes", "source": "tool:<nom_outil>"}],
 "differences": {"features": ["..."], "configs": ["..."]}}
Chaque fait doit reprendre des valeurs renvoyées par un outil, sans rien inventer."""

KNOWLEDGE_SYSTEM = """Tu es l'Agent 2 (Connaissance). Tu cherches dans la documentation interne avec l'outil search_knowledge.
Tu n'as PAS accès aux données du cas. Rassemble les règles générales utiles pour expliquer un écart Batch / Real-Time :
règles de fraîcheur des snapshots, tolérance d'écart, définition des features (sales_7d, promotion, price, stock_level),
versions de modèle et configurations, différence entre symptôme et cause racine, cas de causes multiples,
incidents passés similaires, procédure du runbook.
Fais plusieurs recherches avec des termes concrets. Ne retiens que ce que les documents disent vraiment.

Réponse finale : UNIQUEMENT un objet JSON de la forme :
{"rules": [{"rule": "règle ou définition reprise des documents", "source": "doc:<doc_id>"}],
 "similar_incidents": [{"summary": "...", "resolution": "...", "source": "doc:<doc_id>"}]}"""

ANALYST_SYSTEM = """Tu es l'Agent 3 (Analyste et Vérificateur). Tu reçois les FAITS de l'Agent 1 et les RÈGLES de l'Agent 2.
Tu n'as pas d'outil. Tu dois :
1. Appliquer les règles documentées aux faits observés.
2. Écarter les hypothèses non prouvées. Ne devine pas : chaque preuve doit venir des faits ou des règles fournis.
3. Un symptôme n'est pas une cause racine : si une différence est la conséquence d'une autre, la cause racine est l'autre.
4. Si aucune cause n'est établie par les faits et les règles, la réponse est NO_ISSUE.
5. Choisir UNE cause principale ; les autres causes éventuelles vont dans secondary_causes.
6. Donner une recommandation concrète, reprise du runbook ou d'incidents similaires si ils sont fournis.
7. Citer pour chaque preuve sa source exacte : "tool:<nom_outil>" ou "doc:<doc_id>", telles qu'elles figurent dans les entrées.
8. Rédige les claim et la recommandation en français. Les sources (tool:..., doc:...), les noms de causes et les nombres restent inchangés. Écris les nombres sans séparateur de milliers (1200, pas 1 200).

Causes possibles : DATA_MISMATCH, FEATURE_MISMATCH, CONFIG_MISMATCH, MODEL_VERSION_MISMATCH, STALE_DATA, NO_ISSUE.

Réponse finale : UNIQUEMENT un objet JSON de la forme :
{"primary_cause": "...", "secondary_causes": [],
 "evidence": [{"claim": "...", "source": "tool:... ou doc:..."}],
 "confidence": "high|medium|low", "recommendation": "..."}"""

# COMMAND ----------

# Agents 1 et 2, vérification et normalisation du rapport
def dispatch_data(name, args):
    return call_tool(name, args)


def dispatch_rag(name, args):
    if name != "search_knowledge":
        return json.dumps({"error": "outil non disponible"})
    return json.dumps(search_knowledge(args.get("query"), args.get("k", 3)), ensure_ascii=False, default=str)


def agent1_investigator(case_id):
    r = run_llm_agent(INVESTIGATOR_SYSTEM, f"Collecte les faits pour {case_id}.", list(TOOL_SCHEMAS), dispatch_data,
                      validate=lambda o: isinstance(o.get("facts"), list))
    return r


def agent2_knowledge(case_id):
    if not USE_RAG:
        return {"obj": {"rules": [], "similar_incidents": []}, "raw": "", "repaired": False, "tools": [],
                "tool_outputs": [], "n_llm": 0, "error": None}
    return run_llm_agent(KNOWLEDGE_SYSTEM,
                         "Rassemble les règles et incidents utiles pour enquêter sur un écart entre prévisions Batch et Real-Time.",
                         [SEARCH_KNOWLEDGE_SCHEMA], dispatch_rag,
                         validate=lambda o: isinstance(o.get("rules"), list))


def _nums(s):
    s = str(s).replace("\u202f", "").replace("\u00a0", "")
    out = set()
    for x in re.findall(r"\d+(?:[.,]\d+)?", s):
        try:
            out.add(round(float(x.replace(",", ".")), 2))
        except ValueError:
            pass
    return out


def normalize_report(obj):
    """Retourne un rapport normalisé ou None s'il est inutilisable."""
    if not isinstance(obj, dict) or obj.get("primary_cause") not in CAUSES:
        return None
    sec = obj.get("secondary_causes") or []
    sec = [c for c in sec if c in CAUSES and c != obj["primary_cause"]] if isinstance(sec, list) else []
    ev = []
    for e in (obj.get("evidence") or []):
        ev.append({"claim": str(e.get("claim", "")), "source": e.get("source")} if isinstance(e, dict)
                  else {"claim": str(e), "source": None})
    return {"primary_cause": obj["primary_cause"], "secondary_causes": list(dict.fromkeys(sec)),
            "evidence": ev, "confidence": obj.get("confidence") if obj.get("confidence") in CONFIDENCES else None,
            "recommendation": str(obj.get("recommendation") or "")}


def verify_report(report, inv_res, kn_res):
    """Vérification par le code : sources valides et nombres cités retrouvés dans les entrées."""
    allowed = {f"tool:{n}" for n in inv_res["tools"]}
    pool = set()
    for _, content in inv_res["tool_outputs"]:
        pool |= _nums(content)
    for _, content in kn_res["tool_outputs"]:
        pool |= _nums(content)
        try:
            for x in json.loads(content).get("results", []):
                allowed.add(f"doc:{x['doc_id']}")
        except Exception:
            pass
    problems = []
    if not report["evidence"]:
        problems.append("aucune preuve fournie")
    for e in report["evidence"]:
        if not e["source"]:
            problems.append(f"source manquante : {e['claim'][:80]}")
        elif e["source"] not in allowed:
            problems.append(f"source inconnue {e['source']} : {e['claim'][:60]}")
        missing = _nums(e["claim"]) - pool
        if missing:
            problems.append(f"nombres absents des entrées {sorted(missing)} : {e['claim'][:60]}")
    if not report["recommendation"].strip():
        problems.append("recommandation manquante")
    return problems, sorted(allowed)

# COMMAND ----------

# Agent 3 et graphe LangGraph
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, START, END


class State(TypedDict, total=False):
    case_id: str
    inv: Any
    kn: Any
    final: Any
    analyst_raw: str
    analyst_llm: int
    analyst_repaired: bool


def node_investigator(state):
    return {"inv": agent1_investigator(state["case_id"])}


def node_knowledge(state):
    return {"kn": agent2_knowledge(state["case_id"])}


def node_analyst(state):
    inv, kn = state["inv"], state["kn"]
    inv_obj = inv["obj"] or {"facts": [], "note": "Agent 1 n'a pas produit de faits exploitables"}
    kn_obj = kn["obj"] or {"rules": [], "similar_incidents": [], "note": "Agent 2 n'a pas produit de règles exploitables"}
    rag_note = "" if USE_RAG else "\nAucune documentation n'est disponible : n'invente aucun seuil."
    user = (f"Cas : {state['case_id']}\n\nFAITS (Agent 1) :\n{json.dumps(inv_obj, ensure_ascii=False)}\n\n"
            f"RÈGLES ET INCIDENTS (Agent 2) :\n{json.dumps(kn_obj, ensure_ascii=False)}{rag_note}")
    res = run_llm_agent(ANALYST_SYSTEM, user, None, lambda n, a: "{}",
                        validate=lambda o: o.get("primary_cause") in CAUSES)
    n_llm, report = res["n_llm"], normalize_report(res["obj"])
    verification = {"passed": False, "problems": ["rapport illisible"], "revised": False, "sources_allowed": []}
    if report:
        problems, allowed = verify_report(report, inv, kn)
        revised = False
        if problems:                                   # une seule révision, signalée
            revised = True
            fb = ("Vérification automatique : problèmes détectés.\n- " + "\n- ".join(problems[:8]) +
                  f"\nSources autorisées : {allowed}\nCorrige uniquement ce qui est faux, garde le même format JSON, "
                  "et ne cite que des nombres présents dans les entrées.")
            res2 = run_llm_agent(ANALYST_SYSTEM, user + "\n\n" + fb, None, lambda n, a: "{}",
                                 validate=lambda o: o.get("primary_cause") in CAUSES)
            n_llm += res2["n_llm"]
            report2 = normalize_report(res2["obj"])
            if report2:
                problems2, _ = verify_report(report2, inv, kn)
                report, problems = report2, problems2
        verification = {"passed": not problems, "problems": problems, "revised": revised, "sources_allowed": allowed}
        report["verification"] = verification
    return {"final": report, "analyst_raw": res["raw"], "analyst_llm": n_llm, "analyst_repaired": res["repaired"]}


def build_graph():
    g = StateGraph(State)
    g.add_node("investigator", node_investigator)
    g.add_node("knowledge", node_knowledge)
    g.add_node("analyst", node_analyst)
    if PARALLEL:
        g.add_edge(START, "investigator")
        g.add_edge(START, "knowledge")
        g.add_edge(["investigator", "knowledge"], "analyst")   # l'analyste attend les deux
    else:
        g.add_edge(START, "investigator")
        g.add_edge("investigator", "knowledge")
        g.add_edge("knowledge", "analyst")
    g.add_edge("analyst", END)
    return g.compile()


GRAPH = build_graph()


def run_case(case_id):
    t0 = time.time()
    st = GRAPH.invoke({"case_id": case_id})
    inv, kn = st["inv"], st["kn"]
    return {"case_id": case_id, "report": st.get("final"), "raw": st.get("analyst_raw", ""),
            "inv": inv, "kn": kn,
            "tools": inv["tools"] + kn["tools"],
            "tool_outputs": inv["tool_outputs"] + kn["tool_outputs"],
            "n_llm": inv["n_llm"] + kn["n_llm"] + st.get("analyst_llm", 0),
            "repaired": bool(inv["repaired"] or kn["repaired"] or st.get("analyst_repaired")),
            "error": inv["error"] or kn["error"], "seconds": round(time.time() - t0, 1)}

# COMMAND ----------

# Démonstration : une question, un rapport complet (CASE_009)
r = run_case("CASE_009")
print(json.dumps(r["report"], indent=2, ensure_ascii=False))
print(f"\nAgent 1 : {len(r['inv']['tools'])} outils | Agent 2 : {len(r['kn']['tools'])} recherches | "
      f"appels LLM : {r['n_llm']} | {r['seconds']} s")

# COMMAND ----------

# Exécution des cas (N_RUNS passes). Un cas qui plante est noté, il n'arrête pas la boucle.
BATCH_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
results = []
for run_idx in range(1, N_RUNS + 1):
    print(f"===== passe {run_idx}/{N_RUNS} =====")
    for c in CASES:
        try:
            r = run_case(c)
        except Exception as e:
            r = {"case_id": c, "report": None, "raw": "", "inv": {}, "kn": {}, "tools": [], "tool_outputs": [],
                 "n_llm": 0, "repaired": False, "error": f"{type(e).__name__}: {str(e)[:150]}", "seconds": 0.0}
        r["run"] = run_idx
        results.append(r)
        rep = r["report"] or {}
        ver = rep.get("verification", {})
        flags = ((" | RELANCÉ" if r["repaired"] else "")
                 + (" | RÉVISÉ" if ver.get("revised") else "")
                 + ("" if not rep or ver.get("passed") else " | VÉRIF. ÉCHOUÉE")
                 + (f" | ERREUR {r['error']}" if r["error"] else "")
                 + ("" if rep else " | RAPPORT ILLISIBLE"))
        print(f"{c} -> {rep.get('primary_cause')} {rep.get('secondary_causes', [])} ({rep.get('confidence')}) | "
              f"{r['n_llm']} appels LLM, {r['seconds']} s{flags}")

# COMMAND ----------

# Diagnostic : rapports relancés, révisés, illisibles ou dont la vérification a échoué
bad = [r for r in results if r["repaired"] or r["report"] is None
       or (r["report"].get("verification", {}).get("revised")) or not r["report"].get("verification", {}).get("passed")]
if not bad:
    print("Aucun rapport à signaler.")
for r in bad:
    ver = (r["report"] or {}).get("verification", {})
    print(f"\n--- {r['case_id']} (passe {r['run']}) | relancé : {r['repaired']} | révisé : {ver.get('revised')} | "
          f"vérification réussie : {ver.get('passed')} | erreur : {r['error']}")
    for p in ver.get("problems", [])[:5]:
        print("   problème :", p)
    if r["report"] is None:
        print("   réponse brute de l'Agent 3 :", repr(r["raw"])[:500])

# COMMAND ----------

# ÉVALUATION (cellule réservée : la seule qui lit la vérité terrain)
import statistics
from collections import defaultdict

cat = spark.sql("SELECT current_catalog()").first()[0]
truth_df = spark.table(f"{cat}.forecast_debug.known_issues")
truth, truth_sec = {}, {}
NONE_LIKE = {None, "", "NONE", "None", "none", "null", "NULL", "NO_ISSUE"}
for row in truth_df.collect():
    truth[row["case_id"]] = row["expected_root_cause"]
    truth_sec[row["case_id"]] = row["secondary_cause"]


def expected_set(case_id):
    return {c for c in (truth[case_id], truth_sec.get(case_id)) if c not in NONE_LIKE} or {"NO_ISSUE"}


def predicted_set(rep):
    return set() if not rep else {rep["primary_cause"], *rep.get("secondary_causes", [])}


def _slim(rep):
    """Rapport sans la liste des sources autorisées (longue, inutile dans la table)."""
    if not rep:
        return None
    d = dict(rep)
    v = dict(d.get("verification", {}))
    v.pop("sources_allowed", None)
    d["verification"] = v
    return d


def _outs(r):
    """Sorties d'outils : données de l'Agent 1 en entier, Agent 2 réduit à la liste des documents retrouvés."""
    out = [[n, c] for n, c in (r["inv"] or {}).get("tool_outputs", [])]
    for n, c in (r["kn"] or {}).get("tool_outputs", []):
        try:
            res = json.loads(c).get("results", [])
            c = json.dumps({"results": [{"doc_id": x.get("doc_id"), "section": x.get("section")} for x in res]})
        except Exception:
            c = ""
        out.append([n, c])
    return json.dumps(out, ensure_ascii=False)


config = f"multi{'_rag' if USE_RAG else '_norag'}_{PROMPT_VERSION}_{MODEL}"
by_run, rows = defaultdict(list), []
tp = fp = fn = 0
ver_pass = ver_rev = 0

for r in results:
    cid, rep = r["case_id"], r["report"]
    pred = rep["primary_cause"] if rep else "INVALID"
    good = pred == truth[cid]
    by_run[r["run"]].append(good)
    exp, prd = expected_set(cid), predicted_set(rep)
    tp += len(exp & prd); fp += len(prd - exp); fn += len(exp - prd)
    ver = (rep or {}).get("verification", {})
    ver_pass += bool(ver.get("passed")); ver_rev += bool(ver.get("revised"))
    print(f"{'OK ' if good else 'ERR'} passe {r['run']} {cid} prédit : {pred:24s} | attendu : {truth[cid]}"
          + ("" if exp == prd else f" | ensemble attendu {sorted(exp)} vs prédit {sorted(prd)}"))
    rows.append((BATCH_ID, config, r["run"], cid, pred, json.dumps((rep or {}).get("secondary_causes", [])),
                 str((rep or {}).get("confidence")), len(r["tools"]), ",".join(r["tools"]), r["n_llm"],
                 float(r["seconds"]), bool(r["repaired"]), bool(good),
                 json.dumps(_slim(rep), ensure_ascii=False)[:8000] if rep else (r["raw"] or "")[:2000],
                 "", _outs(r),
                 datetime.now(timezone.utc).isoformat()))

accs = [sum(v) / len(v) for v in by_run.values()]
n = len(results)
print("\n=== RÉSUMÉ", config, "===")
print("Exactitude cause principale par passe :", [f"{sum(v)}/{len(v)}" for v in by_run.values()])
print(f"Moyenne : {100 * statistics.mean(accs):.1f} %"
      + (f" | min {100 * min(accs):.1f} % | max {100 * max(accs):.1f} %" if len(accs) > 1 else ""))
if tp + fp and tp + fn:
    print(f"Ensemble de causes : précision {tp / (tp + fp):.2f} | rappel {tp / (tp + fn):.2f}")
print(f"Appels LLM moyens : {sum(r['n_llm'] for r in results) / n:.1f} | temps moyen : {sum(r['seconds'] for r in results) / n:.1f} s")
print(f"Vérification du code réussie : {ver_pass}/{n} | rapports révisés : {ver_rev}/{n} | "
      f"relances de format : {sum(r['repaired'] for r in results)}/{n} | illisibles : {sum(r['report'] is None for r in results)}/{n}")

if N_RUNS > 1:
    print("\nStabilité par cas :")
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
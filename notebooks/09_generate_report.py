# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 09 - Génération dynamique du rapport d'évaluation
# MAGIC
# MAGIC Ce notebook **relit les résultats réels** (table `agent_runs_v2`, vérité terrain `known_issues`, et si elle existe `rule_baseline_runs`) et **recalcule** tous les chiffres à chaque exécution : exactitude, intervalles de confiance, précision/rappel, coût, stabilité, erreurs par cas, effet du RAG si les deux configurations ont été mesurées.
# MAGIC Les conclusions et les limites sont **déduites des chiffres** (par exemple : « les intervalles se recouvrent », « une seule passe », « effet du RAG non mesuré »). Rien n'est écrit en dur sur les résultats.
# MAGIC
# MAGIC **Sortie** : un rapport Markdown, affiché dans le notebook, sauvegardé dans le volume (`/Volumes/<catalog>/forecast_debug/raw/reports/`) et dans la table `evaluation_reports`.
# MAGIC
# MAGIC Ce notebook lit `known_issues` : c'est un notebook d'évaluation, il ne doit jamais être appelé par un agent.
# MAGIC
# MAGIC **Pour inclure la baseline par règles**, ajoutez à la fin de `03_rule_based_baseline` la cellule donnée dans le dernier commentaire de ce notebook.

# COMMAND ----------

import shutil, os

src = "/Volumes/workspace/forecast_debug/raw/reports/RAPPORT_EVALUATION_latest.md"
dst = "/Workspace/Users/imen.kammoun@enis.tn/Forecast-Debugging-Agent/RAPPORT_EVALUATION.md"

shutil.copyfile(src, dst)
print("copié :", os.path.exists(dst))

# COMMAND ----------

# Configuration
SELECTION = "latest_per_config"   # "latest_per_config" : le dernier batch de chaque configuration
                                  # "all"               : tous les batches de la table
BATCH_IDS = []                    # ou une liste précise, par exemple ["20261118_210216_db50df", "20261118_212936_875073"]
REFERENCE_SYSTEM = "single"       # système de référence pour les ratios de coût ("single" ou "multi")
INCLUDE_BASELINE = True           # lit rule_baseline_runs si la table existe
SAVE_REPORT = True
BREAKDOWN_COLS = ["difficulty", "requires_rag", "split"]   # ventilations utilisées si la colonne existe dans known_issues
TEAM_NOTES = ""                   # interprétation écrite à la main par l'équipe (hypothèses, explications), ajoutée en section 10

# COMMAND ----------

# Chargement des données
cat = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA_FQ = f"{cat}.forecast_debug"

runs_pdf = spark.table(f"{SCHEMA_FQ}.agent_runs_v2").toPandas()
truth_pdf = spark.table(f"{SCHEMA_FQ}.known_issues").toPandas()

baseline_pdf = None
if INCLUDE_BASELINE:
    try:
        baseline_pdf = spark.table(f"{SCHEMA_FQ}.rule_baseline_runs").toPandas()
    except Exception:
        baseline_pdf = None

print(f"agent_runs_v2 : {len(runs_pdf)} lignes, {runs_pdf['batch_id'].nunique()} batches")
print(f"known_issues  : {len(truth_pdf)} cas, colonnes {list(truth_pdf.columns)}")
print("rule_baseline_runs :", "trouvée" if baseline_pdf is not None else "absente (la baseline sera omise)")
print(runs_pdf.groupby(["batch_id", "config"]).agg(passes=("run", "nunique"), lignes=("case_id", "count"),
                                                   ts=("ts", "max")).sort_values("ts").to_string())

# COMMAND ----------

# Fonctions utilitaires (calcul pur, sans Spark)
import json, re, math, html
from collections import Counter
from datetime import datetime, timezone

NONE_LIKE = {None, "", "NONE", "None", "none", "null", "NULL", "NO_ISSUE"}
SYSTEM_NAMES = {"single": "Agent unique", "multi": "3 agents", "rules": "Règles (sans LLM)"}


def _clean(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def fr(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return "n/a" if x is None else fr(100 * x, d) + " %"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test_p(b, c):
    """Test des signes exact, bilatéral, sur les cas discordants."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _nums(s):
    s = str(s).replace("\u202f", "").replace("\u00a0", "")
    out = set()
    for x in re.findall(r"\d+(?:[.,]\d+)?", s):
        try:
            out.add(round(float(x.replace(",", ".")), 2))
        except ValueError:
            pass
    return out


def parse_config(config):
    """single_rag_v1_databricks-gpt-oss-120b -> système, RAG, prompt, modèle."""
    parts = str(config).split("_")
    if len(parts) >= 4 and parts[0] in ("single", "multi") and parts[1] in ("rag", "norag"):
        return {"system": parts[0], "rag": parts[1], "prompt": "_".join(parts[2:-1]), "model": parts[-1]}
    return {"system": "other", "rag": "?", "prompt": "?", "model": str(config)}


def make_label(info):
    name = SYSTEM_NAMES.get(info["system"], info["system"])
    rag = "" if info["rag"] == "?" else (" + RAG" if info["rag"] == "rag" else " sans RAG")
    prompt = "" if info["prompt"] == "?" else f" [{info['prompt']}]"
    return f"{name}{rag}{prompt}"


def parse_json_obj(text):
    text = _clean(text) or ""
    try:
        s, e = text.index("{"), text.rindex("}")
        obj = json.loads(text[s:e + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_list(s):
    s = _clean(s)
    if not s:
        return []
    try:
        v = json.loads(s)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []


def evidence_texts(rep):
    if not rep:
        return []
    out = []
    for e in (rep.get("evidence") or []):
        out.append(str(e.get("claim", "")) if isinstance(e, dict) else str(e))
    return out


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def make_truth(truth_pdf, breakdown_cols):
    cols = set(truth_pdf.columns)
    out = {}
    for _, r in truth_pdf.iterrows():
        prim = _clean(r["expected_root_cause"])
        sec = _clean(r["secondary_cause"]) if "secondary_cause" in cols else None
        expected = {c for c in (prim, sec) if c not in NONE_LIKE} or {"NO_ISSUE"}
        out[r["case_id"]] = {
            "primary": prim, "expected": expected,
            "desc": str(_clean(r["issue_description"])) if "issue_description" in cols and _clean(r["issue_description"]) else "",
            "attrs": {c: str(_clean(r[c])) for c in breakdown_cols if c in cols}}
    return out

# COMMAND ----------

# Calcul des métriques pour un ensemble de lignes (un batch, ou la baseline)
def compute_metrics(df, truth, info, batch_id=None, ts=None):
    rows, skipped, trunc = [], 0, 0
    for _, r in df.iterrows():
        cid = r["case_id"]
        if cid not in truth:
            skipped += 1
            continue
        t = truth[cid]
        pred = _clean(r["predicted"]) or "INVALID"
        sec = [c for c in parse_list(r["secondary"] if "secondary" in df.columns else None) if c != pred]
        pset = set() if pred == "INVALID" else {pred, *sec}
        rep = parse_json_obj(r["raw"]) if "raw" in df.columns else None
        tool_out = (_clean(r["tool_outputs"]) if "tool_outputs" in df.columns else None) or ""
        if len(tool_out) >= 11990:
            trunc += 1
        ev = evidence_texts(rep)
        grounding = None
        if ev and tool_out:
            cited = set().union(*[_nums(e) for e in ev])
            grounding = (len(cited & _nums(tool_out)), len(cited))
        ver = rep.get("verification") if rep and isinstance(rep.get("verification"), dict) else None
        first_raw = (_clean(r["first_raw"]) if "first_raw" in df.columns else None) or ""
        repaired = bool(_clean(r["repaired"])) if "repaired" in df.columns else False
        rows.append({
            "run": int(_clean(r["run"]) or 1) if "run" in df.columns else 1, "case": cid, "pred": pred, "sec": sec,
            "pset": pset, "correct": pred == t["primary"], "exact": pset == t["expected"],
            "repaired": repaired, "empty_first": repaired and info["system"] == "single" and not first_raw.strip(),
            "n_llm": float(_clean(r["n_llm"]) or 0) if "n_llm" in df.columns else 0.0,
            "seconds": float(_clean(r["seconds"]) or 0) if "seconds" in df.columns else 0.0,
            "n_tools": float(_clean(r["n_tools"]) or 0) if "n_tools" in df.columns else 0.0,
            "grounding": grounding, "ver": ver,
            "parsed": rep is not None, "reco": bool(rep and str(rep.get("recommendation") or "").strip())})

    m = {"info": info, "label": make_label(info), "batch_id": batch_id, "ts": ts, "rows": rows,
         "skipped": skipped, "trunc": trunc}
    n_rows = len(rows)
    m["n_rows"] = n_rows
    runs = sorted({x["run"] for x in rows})
    m["passes"] = len(runs)
    cases = sorted({x["case"] for x in rows})
    m["cases"] = cases
    m["n_cases"] = len(cases)
    m["per_run"] = []
    for run in runs:
        rr = [x for x in rows if x["run"] == run]
        m["per_run"].append((run, sum(x["correct"] for x in rr), len(rr)))
    accs = [k / n for _, k, n in m["per_run"] if n]
    m["mean_acc"] = sum(accs) / len(accs) if accs else None
    m["min_acc"], m["max_acc"] = (min(accs), max(accs)) if accs else (None, None)
    m["ci"] = wilson(round((m["mean_acc"] or 0) * m["n_cases"]), m["n_cases"]) if accs else (None, None)

    tp = fp = fn = 0
    for x in rows:
        exp = truth[x["case"]]["expected"]
        tp += len(exp & x["pset"]); fp += len(x["pset"] - exp); fn += len(exp - x["pset"])
    m["precision"] = tp / (tp + fp) if tp + fp else None
    m["recall"] = tp / (tp + fn) if tp + fn else None

    def avg(key):
        return sum(x[key] for x in rows) / n_rows if n_rows else None
    m["avg_llm"], m["avg_sec"], m["avg_tools"] = avg("n_llm"), avg("seconds"), avg("n_tools")
    m["repaired_n"] = sum(x["repaired"] for x in rows)
    m["empty_n"] = sum(x["empty_first"] for x in rows)
    m["invalid_n"] = sum(x["pred"] == "INVALID" for x in rows)
    g = [x["grounding"] for x in rows if x["grounding"]]
    m["grounding"] = (sum(a for a, _ in g), sum(b for _, b in g)) if g else None
    vers = [x["ver"] for x in rows if x["ver"]]
    m["ver"] = (sum(bool(v.get("passed")) for v in vers), sum(bool(v.get("revised")) for v in vers),
                len(vers), n_rows - len(vers)) if vers else None
    parsed = [x for x in rows if x["parsed"]]
    m["reco"] = (sum(x["reco"] for x in parsed), len(parsed)) if parsed else None

    m["case_n"], m["case_frac"], m["case_preds"], m["case_psets"], m["case_exact"] = {}, {}, {}, {}, {}
    for c in cases:
        cr = [x for x in rows if x["case"] == c]
        m["case_n"][c] = len(cr)
        m["case_frac"][c] = sum(x["correct"] for x in cr) / len(cr)
        m["case_preds"][c] = Counter(x["pred"] for x in cr)
        m["case_psets"][c] = [x["pset"] for x in cr]
        m["case_exact"][c] = [x["exact"] for x in cr]
    return m

# COMMAND ----------

# Sélection des batches et construction des métriques
def build_metrics(runs, truth, baseline):
    batches = (runs.groupby("batch_id").agg(config=("config", "first"), ts=("ts", "max")).reset_index())
    batches["info"] = batches["config"].map(parse_config)
    if BATCH_IDS:
        chosen = batches[batches["batch_id"].isin(BATCH_IDS)].copy()
        chosen["order"] = chosen["batch_id"].map({b: i for i, b in enumerate(BATCH_IDS)})
        chosen = chosen.sort_values("order")
    elif SELECTION == "all":
        chosen = batches.sort_values("ts")
    else:
        batches["key"] = batches["info"].map(lambda i: (i["system"], i["rag"], i["prompt"], i["model"]))
        chosen = batches.sort_values("ts").groupby("key").tail(1).sort_values("ts")
    metrics = []
    if baseline is not None and len(baseline):
        b = baseline.copy()
        b["run"] = 1
        metrics.append(compute_metrics(b, truth, {"system": "rules", "rag": "?", "prompt": "?", "model": "n/a"},
                                       batch_id="rule_baseline_runs", ts=str(b["ts"].max()) if "ts" in b else None))
    labels = Counter()
    for _, row in chosen.iterrows():
        m = compute_metrics(runs[runs["batch_id"] == row["batch_id"]], truth, row["info"], row["batch_id"], row["ts"])
        labels[m["label"]] += 1
        metrics.append(m)
    seen = Counter()
    for m in metrics:                       # libellés uniques si plusieurs batches de même configuration
        if labels[m["label"]] > 1:
            seen[m["label"]] += 1
            m["label"] += f" ({str(m['batch_id'])[-6:]})"
    return metrics, batches

# COMMAND ----------

# Sections du rapport
def sec_results(metrics):
    H = ["Mesure"] + [m["label"] for m in metrics]
    def row(name, f):
        return [name] + [f(m) for m in metrics]
    rows = [
        row("Passes", lambda m: m["passes"]),
        row("Cas évalués", lambda m: m["n_cases"]),
        row("Cause principale par passe", lambda m: " ; ".join(f"{k}/{n}" for _, k, n in m["per_run"])),
        row("**Exactitude moyenne**", lambda m: "**" + pct(m["mean_acc"]) + "**"),
        row("Min – max", lambda m: f"{pct(m['min_acc'])} – {pct(m['max_acc'])}" if m["passes"] > 1 else "une seule passe"),
        row("Intervalle de confiance 95 % (Wilson)", lambda m: f"[{pct(m['ci'][0], 0)} ; {pct(m['ci'][1], 0)}]"),
        row("Précision (ensemble de causes)", lambda m: fr(m["precision"], 2)),
        row("Rappel (ensemble de causes)", lambda m: fr(m["recall"], 2)),
        row("Appels LLM par cas", lambda m: fr(m["avg_llm"])),
        row("Temps moyen par cas (s)", lambda m: fr(m["avg_sec"])),
        row("Appels d'outils par cas", lambda m: fr(m["avg_tools"]) if m["info"]["system"] != "rules" else "n/a"),
        row("Relances de format", lambda m: f"{m['repaired_n']}/{m['n_rows']}" if m["info"]["system"] != "rules" else "n/a"),
        row("  dont réponse initiale vide (agent unique)", lambda m: m["empty_n"] if m["info"]["system"] == "single" else "n/a"),
        row("Réponses illisibles", lambda m: f"{m['invalid_n']}/{m['n_rows']}"),
        row("Nombres cités retrouvés dans les sorties d'outils", lambda m: f"{m['grounding'][0]}/{m['grounding'][1]}" if m["grounding"] else "n/a"),
        row("Vérification par le code (réussie / révisés)", lambda m: (f"{m['ver'][0]}/{m['ver'][2]} / {m['ver'][1]}" + (f" (+{m['ver'][3]} non lisibles)" if m["ver"][3] else "") if m["ver"] else "n/a")),
        row("Recommandation fournie", lambda m: f"{m['reco'][0]}/{m['reco'][1]}" if m["reco"] else "n/a"),
    ]
    return md_table(H, rows)


def sec_history(batches, truth):
    rows = []
    for _, b in batches.sort_values("ts").iterrows():
        info = b["info"]
        m = BATCH_CACHE.get(b["batch_id"])
        if m is None:
            m = compute_metrics(RUNS[RUNS["batch_id"] == b["batch_id"]], truth, info, b["batch_id"], b["ts"])
            BATCH_CACHE[b["batch_id"]] = m
        rows.append([str(b["ts"])[:19], b["batch_id"], make_label(info), m["passes"], m["n_rows"], pct(m["mean_acc"])])
    return md_table(["Date (UTC)", "batch_id", "Configuration", "Passes", "Lignes", "Exactitude moyenne"], rows)


def case_cell(m, cid, truth):
    if cid not in m["case_frac"]:
        return "n/a"
    n = m["case_n"][cid]
    wrong = n - round(m["case_frac"][cid] * n)
    if wrong == 0:
        return "OK" if all(m["case_exact"][cid]) else "OK, causes secondaires différentes"
    preds = ", ".join(f"{p} x{c}" for p, c in m["case_preds"][cid].items() if p != truth[cid]["primary"])
    return ("INSTABLE : " if wrong < n else "") + f"{wrong}/{n} faux ({preds})"


def sec_per_case(metrics, truth):
    cases = sorted(truth)
    rows = [[c, "+".join(sorted(truth[c]["expected"]))] + [case_cell(m, c, truth) for m in metrics] for c in cases]
    return md_table(["Cas", "Causes attendues"] + [m["label"] for m in metrics], rows)


def sec_errors(metrics, truth):
    out = []
    for cid in sorted(truth):
        findings = []
        for m in metrics:
            if cid not in m["case_frac"]:
                continue
            n = m["case_n"][cid]
            wrong = n - round(m["case_frac"][cid] * n)
            exp = truth[cid]["expected"]
            psets = m["case_psets"][cid]
            extras, missing = Counter(), Counter()
            for ps in psets:
                if m["info"]["system"] == "rules":
                    continue
                for c in ps - exp:
                    extras[c] += 1
                for c in exp - ps:
                    missing[c] += 1
            notes = []
            if wrong == n:
                notes.append(f"erreur stable ({wrong}/{n} passes fausses)")
            elif wrong > 0:
                notes.append(f"instable ({wrong}/{n} passes fausses)")
            if wrong > 0 and len(exp) > 1 and any(p in exp for p in m["case_preds"][cid]):
                notes.append("la cause principale prédite fait partie des causes attendues : possible ambiguïté sur l'ordre")
            if extras and wrong == 0:
                notes.append("cause(s) en trop : " + ", ".join(f"{c} ({k}/{n} passes)" for c, k in extras.items()))
            elif extras:
                notes.append("autre(s) cause(s) prédite(s) : " + ", ".join(f"{c} ({k}/{n})" for c, k in extras.items()))
            if missing and wrong == 0:
                notes.append("cause attendue absente : " + ", ".join(f"{c} ({k}/{n} passes)" for c, k in missing.items()))
            if notes:
                dist = ", ".join(f"{p} x{c}" for p, c in m["case_preds"][cid].items())
                findings.append(f"- **{m['label']}** : cause principale prédite {dist}. " + " ; ".join(notes) + ".")
        if findings:
            t = truth[cid]
            attrs = ", ".join(f"{k} = {v}" for k, v in t["attrs"].items())
            head = f"#### {cid} : attendu {' + '.join(sorted(t['expected']))}" + (f" ({attrs})" if attrs else "")
            out.append(head + ("\n" + t["desc"] if t["desc"] else "") + "\n\n" + "\n".join(findings))
    return "\n\n".join(out) if out else "Aucune erreur observée dans les configurations sélectionnées."


def sec_empty(metrics):
    items = [f"- {m['label']} : {x['case']} (passe {x['run']})" for m in metrics for x in m["rows"] if x["empty_first"]]
    return "\n".join(items) if items else "Aucune réponse initiale vide."


def sec_cost(metrics):
    llm = [m for m in metrics if m["info"]["system"] != "rules"]
    if not llm:
        return "Aucune configuration à LLM sélectionnée."
    ref = next((m for m in llm if m["info"]["system"] == REFERENCE_SYSTEM), llm[0])
    rows = []
    for m in llm:
        r_llm = m["avg_llm"] / ref["avg_llm"] if ref["avg_llm"] else None
        r_sec = m["avg_sec"] / ref["avg_sec"] if ref["avg_sec"] else None
        rows.append([m["label"], fr(m["avg_llm"]), fr(m["avg_sec"]), fr(m["avg_tools"]),
                     "référence" if m is ref else f"x {fr(r_llm, 2)}", "référence" if m is ref else f"x {fr(r_sec, 2)}"])
    return (md_table(["Configuration", "Appels LLM / cas", "Temps / cas (s)", "Outils / cas",
                      f"Appels vs {ref['label']}", f"Temps vs {ref['label']}"], rows)
            + "\n\nLes jetons consommés ne sont pas enregistrés : le nombre d'appels est un indicateur, pas le coût réel.")


def sec_pairwise(metrics, truth):
    if len(metrics) < 2:
        return "Une seule configuration : pas de comparaison."
    rows = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            a, b = metrics[i], metrics[j]
            common = [c for c in a["case_frac"] if c in b["case_frac"]]
            ba = sum(a["case_frac"][c] > b["case_frac"][c] for c in common)
            bb = sum(b["case_frac"][c] > a["case_frac"][c] for c in common)
            rows.append([a["label"], b["label"], len(common), ba, bb, fr(sign_test_p(ba, bb), 3)])
    return (md_table(["Configuration A", "Configuration B", "Cas communs", "Cas où A fait mieux", "Cas où B fait mieux", "p (test des signes)"], rows)
            + "\n\nUn cas « fait mieux » si sa fraction de passes correctes est plus élevée. Une valeur p élevée signifie que la différence n'est pas distinguable du hasard.")


def sec_breakdown(metrics, truth):
    out = []
    for col in BREAKDOWN_COLS:
        values = sorted({t["attrs"][col] for t in truth.values() if col in t["attrs"]})
        if not values:
            continue
        rows = []
        for v in values:
            cs = [c for c, t in truth.items() if t["attrs"].get(col) == v]
            row = [f"{col} = {v}", len(cs)]
            for m in metrics:
                fr_ = [m["case_frac"][c] for c in cs if c in m["case_frac"]]
                row.append(pct(sum(fr_) / len(fr_)) if fr_ else "n/a")
            rows.append(row)
        out.append(md_table(["Groupe", "Cas"] + [m["label"] for m in metrics], rows))
    return "\n\n".join(out) if out else "Aucune colonne de ventilation (difficulty, requires_rag, split) dans known_issues."


def sec_rag_effect(metrics):
    lines = []
    groups = {}
    for m in metrics:
        i = m["info"]
        if i["system"] in ("single", "multi"):
            groups.setdefault((i["system"], i["prompt"], i["model"]), {})[i["rag"]] = m
    for (system, prompt, model), d in groups.items():
        if "rag" in d and "norag" in d:
            a, b = d["rag"], d["norag"]
            lines.append(f"- {SYSTEM_NAMES[system]} [{prompt}] : avec RAG {pct(a['mean_acc'])} ({a['passes']} passe(s)), "
                         f"sans RAG {pct(b['mean_acc'])} ({b['passes']} passe(s)), écart {fr(100 * (a['mean_acc'] - b['mean_acc']))} points.")
    return "\n".join(lines) if lines else None

# COMMAND ----------

# Conclusions et limites déduites des chiffres
def sec_conclusion(metrics, truth):
    llm = [m for m in metrics if m["info"]["system"] != "rules"]
    L = []
    for m in metrics:
        L.append(f"- **{m['label']}** : {pct(m['mean_acc'])} de cause principale correcte, intervalle de confiance 95 % "
                 f"[{pct(m['ci'][0], 0)} ; {pct(m['ci'][1], 0)}], sur {m['n_cases']} cas et {m['passes']} passe(s).")
    overlaps = all(not (metrics[i]["ci"][1] < metrics[j]["ci"][0] or metrics[j]["ci"][1] < metrics[i]["ci"][0])
                   for i in range(len(metrics)) for j in range(i + 1, len(metrics)))
    if len(metrics) > 1:
        L.append("- Les intervalles de confiance de toutes les configurations **se recouvrent** : les écarts d'exactitude ne sont pas statistiquement significatifs."
                 if overlaps else "- Au moins deux configurations ont des intervalles de confiance qui **ne se recouvrent pas** (voir le tableau) : l'écart y est probablement réel, sous réserve du faible nombre de cas.")
    stable_wrong = [c for c in sorted(truth) if llm and all(c in m["case_frac"] and m["case_frac"][c] == 0 for m in llm)]
    if stable_wrong:
        L.append(f"- Cas toujours faux pour **toutes** les configurations à LLM : {', '.join(stable_wrong)}.")
    unstable = sorted({(m["label"], c) for m in llm for c in m["case_frac"] if 0 < m["case_frac"][c] < 1})
    if unstable:
        L.append("- Cas instables entre passes : " + "; ".join(f"{c} ({lab})" for lab, c in unstable) + ".")
    if len(llm) > 1:
        ref = next((m for m in llm if m["info"]["system"] == REFERENCE_SYSTEM), llm[0])
        for m in llm:
            if m is not ref and ref["avg_llm"] and ref["avg_sec"]:
                L.append(f"- Coût de **{m['label']}** par rapport à **{ref['label']}** : x {fr(m['avg_llm'] / ref['avg_llm'], 2)} appels LLM, x {fr(m['avg_sec'] / ref['avg_sec'], 2)} en temps.")
    for m in llm:
        if m["ver"]:
            L.append(f"- Vérification par le code, **{m['label']}** : {m['ver'][0]}/{m['ver'][2]} rapports conformes, {m['ver'][1]} révisés. "
                     "Un rapport conforme peut contenir une mauvaise cause : le contrôle porte sur les sources et les nombres, pas sur le raisonnement.")
    rag = sec_rag_effect(metrics)
    if rag:
        L.append("- Effet du RAG mesuré :\n" + "\n".join("  " + l for l in rag.splitlines()))
    return "\n".join(L)


def sec_limits(metrics, truth):
    llm = [m for m in metrics if m["info"]["system"] != "rules"]
    n_cases = len(truth)
    L = [f"- **{n_cases} cas** : une erreur change l'exactitude de {fr(100 / n_cases)} points." + (" Échantillon trop petit pour conclure." if n_cases < 30 else "")]
    one = [m["label"] for m in llm if m["passes"] == 1]
    if one:
        L.append("- **Une seule passe** pour : " + ", ".join(one) + ". Leur stabilité n'est pas connue.")
    multi = [m["label"] for m in llm if m["passes"] > 1]
    if multi:
        L.append("- Les passes multiples (" + ", ".join(multi) + ") portent sur les mêmes cas : elles mesurent la stabilité, pas la généralisation.")
    if any(m["info"]["system"] == "rules" for m in metrics):
        L.append("- Les **règles** ont été écrites en connaissant ces cas : leur score ne mesure pas la généralisation et n'est pas comparable à celui des agents.")
    else:
        L.append("- La baseline par règles n'est pas incluse (table `rule_baseline_runs` absente).")
    if not sec_rag_effect(metrics):
        L.append("- **Effet du RAG non mesuré** : aucun couple de configurations avec et sans RAG pour un même système.")
    models = sorted({m["info"]["model"] for m in llm})
    if len(models) == 1:
        L.append(f"- **Un seul modèle** ({models[0]}) : aucune comparaison entre LLM.")
    if any("split" in t["attrs"] for t in truth.values()):
        L.append("- Une colonne `split` existe : vérifiez que les prompts n'ont pas été réglés sur les cas de test.")
    else:
        L.append("- **Pas de séparation développement / test** dans `known_issues` : les erreurs observées ne doivent pas servir à régler un prompt sans nouveaux cas.")
    trunc = [m["label"] for m in metrics if m["trunc"]]
    if trunc:
        L.append("- Les sorties d'outils sont tronquées à 12 000 caractères dans la table pour : " + ", ".join(trunc) + ". Le contrôle des nombres cités peut sous-estimer l'ancrage.")
    skipped = [m["label"] for m in metrics if m["skipped"]]
    if skipped:
        L.append("- Lignes ignorées car le cas n'existe pas dans `known_issues` : " + ", ".join(skipped) + ".")
    L += ["- **Données synthétiques** : aucune conclusion sur la production.",
          "- Les étiquettes peuvent être ambiguës (par exemple deux causes simultanées sans règle d'ordre, ou notions qui se recouvrent) : une « erreur » n'est pas toujours une faute du modèle.",
          "- Les **jetons** consommés ne sont pas mesurés.",
          "- Aucune évaluation humaine de la qualité des preuves n'est incluse."]
    return "\n".join(L)

# COMMAND ----------

# Assemblage du rapport
def build_report(runs, truth_pdf, baseline):
    global RUNS, BATCH_CACHE
    RUNS, BATCH_CACHE = runs, {}
    truth = make_truth(truth_pdf, BREAKDOWN_COLS)
    metrics, batches = build_metrics(runs, truth, baseline)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Rapport d'évaluation : Forecast Debugging Agent",
        f"*Rapport généré automatiquement le {now} à partir de `agent_runs_v2` et `known_issues`. "
        "Tous les chiffres sont recalculés à chaque exécution.*",
        "> **Avertissement.** Les mesures portent sur des cas synthétiques. Elles montrent que le système fonctionne sur des scénarios contrôlés, pas qu'il fonctionnerait sur de vraies données.",
        "## 1. Configurations évaluées",
        md_table(["Configuration", "Source", "Passes", "Lignes"],
                 [[m["label"], m["batch_id"], m["passes"], m["n_rows"]] for m in metrics]),
        "## 2. Résultats globaux", sec_results(metrics),
        "## 3. Résultat par cas", sec_per_case(metrics, truth),
        "## 4. Ventilation par groupe de cas", sec_breakdown(metrics, truth),
        "## 5. Analyse d'erreurs (générée à partir des prédictions)", sec_errors(metrics, truth),
        "### Réponses initiales vides du modèle", sec_empty(metrics),
        "## 6. Coût", sec_cost(metrics),
        "## 7. Comparaisons deux à deux", sec_pairwise(metrics, truth),
        "## 8. Conclusions déduites des chiffres", sec_conclusion(metrics, truth),
        "## 9. Limites", sec_limits(metrics, truth),
    ]
    if TEAM_NOTES.strip():
        parts += ["## 10. Interprétation de l'équipe (écrite à la main, non calculée)", TEAM_NOTES.strip()]
    parts += ["## Annexe : tous les runs de la table", sec_history(batches, truth)]
    return "\n\n".join(parts) + "\n", metrics

# COMMAND ----------

# Génération, affichage et sauvegarde
report_md, metrics = build_report(runs_pdf, truth_pdf, baseline_pdf)

try:
    import markdown
    displayHTML(markdown.markdown(report_md, extensions=["tables", "fenced_code"]))
except Exception:
    displayHTML("<pre style='white-space:pre-wrap'>" + html.escape(report_md) + "</pre>")

if SAVE_REPORT:
    import os
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = f"/Volumes/{cat}/forecast_debug/raw/reports"
    for name in (f"RAPPORT_EVALUATION_{stamp}.md", "RAPPORT_EVALUATION_latest.md"):
        path = f"{folder}/{name}"
        try:
            os.makedirs(folder, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report_md)
        except Exception:
            dbutils.fs.put(path, report_md, True)
        print("Rapport écrit :", path)
    try:
        spark.createDataFrame(
            [(stamp, datetime.now(timezone.utc).isoformat(), SELECTION if not BATCH_IDS else "BATCH_IDS",
              json.dumps([m["batch_id"] for m in metrics]),
              json.dumps({m["label"]: {"accuracy": m["mean_acc"], "passes": m["passes"], "n_cases": m["n_cases"],
                                       "precision": m["precision"], "recall": m["recall"],
                                       "avg_llm": m["avg_llm"], "avg_sec": m["avg_sec"]} for m in metrics}),
              report_md)],
            "report_id string, ts string, selection string, batch_ids string, metrics_json string, markdown string"
        ).write.mode("append").saveAsTable(f"{cat}.forecast_debug.evaluation_reports")
        print(f"Aussi enregistré dans {cat}.forecast_debug.evaluation_reports")
    except Exception as e:
        print("Table evaluation_reports non écrite :", type(e).__name__, str(e)[:150])
    print("Pour télécharger : Catalog > workspace > forecast_debug > raw > reports > RAPPORT_EVALUATION_latest.md")

# COMMAND ----------

# Cellule à AJOUTER À LA FIN de 03_rule_based_baseline, pour que la baseline apparaisse dans le rapport.
# (suppose que investigate_case(case_id) renvoie un dict avec "primary_cause" et "secondary_causes")
#
# import json
# from datetime import datetime, timezone
# rows = []
# for i in range(1, 13):
#     cid = f"CASE_{i:03d}"
#     r = investigate_case(cid)
#     rows.append((cid, r["primary_cause"], json.dumps(r.get("secondary_causes", [])),
#                  datetime.now(timezone.utc).isoformat()))
# cat = spark.sql("SELECT current_catalog()").first()[0]
# spark.createDataFrame(rows, "case_id string, predicted string, secondary string, ts string") \
#      .write.mode("overwrite").saveAsTable(f"{cat}.forecast_debug.rule_baseline_runs")
# print("rule_baseline_runs écrite")
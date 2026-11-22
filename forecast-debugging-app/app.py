"""Forecast Debugging Agent - frontend (Streamlit, Databricks Apps).

Lit les résultats du système à 3 agents dans Unity Catalog (table agent_runs_v2) et les présente :
vue d'ensemble, détail d'un cas, architecture, évaluation et limites.
Sans DATABRICKS_WAREHOUSE_ID, l'application démarre en mode démonstration avec des données FICTIVES.
"""
import html
import json
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

SCHEMA = os.getenv("FD_SCHEMA", "workspace.forecast_debug")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
DEMO = not WAREHOUSE_ID
STALE_HOURS = 24

CAUSE_TEXT = {
    "STALE_DATA": "Données obsolètes : le snapshot du Batch est trop ancien",
    "DATA_MISMATCH": "Les deux pipelines ont lu des données différentes",
    "FEATURE_MISMATCH": "Une variable d'entrée (promotion, prix...) diffère",
    "CONFIG_MISMATCH": "Une configuration (horizon, prétraitement...) diffère",
    "MODEL_VERSION_MISMATCH": "Les deux pipelines n'utilisent pas la même version du modèle",
    "NO_ISSUE": "Aucun problème : écart dans la tolérance",
}
CONF_ICON = {"high": "🟢 haute", "medium": "🟡 moyenne", "low": "🔴 faible"}
CONF_RANK = {"low": 0, "medium": 1, "high": 2}

st.set_page_config(page_title="Forecast Debugging Agent", page_icon="🔎", layout="wide")

CAUSE_COLOR = {"STALE_DATA": "#F59E0B", "DATA_MISMATCH": "#3B82F6", "FEATURE_MISMATCH": "#8B5CF6",
               "CONFIG_MISMATCH": "#EC4899", "MODEL_VERSION_MISMATCH": "#EF4444", "NO_ISSUE": "#10B981"}
CONF_COLOR = {"high": "#10B981", "medium": "#F59E0B", "low": "#EF4444"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #1E1B4B 0%, #4F46E5 60%, #06B6D4 100%); color: #fff;
       padding: 28px 32px; border-radius: 18px; margin-bottom: 14px;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: 6px 0 0 0; font-size: 1.02rem; opacity: .92;}
.notice {background: #FFF7E6; border: 1px solid #FCD9A0; color: #7A4B00; border-radius: 10px;
         padding: 8px 14px; font-size: .86rem; margin-bottom: 14px;}
.kpi {background: #fff; border-radius: 14px; padding: 16px 18px; border: 1px solid #E5E7EB;
      box-shadow: 0 1px 3px rgba(17,24,39,.06);}
.kpi .v {font-size: 2rem; font-weight: 700; line-height: 1.1;}
.kpi .l {color: #6B7280; font-size: .85rem; margin-top: 2px;}
.chip {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
       color: #fff; margin-right: 6px;}
.case {background: #fff; border: 1px solid #E5E7EB; border-left: 6px solid #9CA3AF; border-radius: 12px;
       padding: 12px 16px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(17,24,39,.05);}
.case .t {font-weight: 700; margin-bottom: 4px;}
.case .r {color: #374151; font-size: .9rem; margin-top: 6px;}
.ev {background: #fff; border: 1px solid #E5E7EB; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;}
.ev .s {color: #6B7280; font-size: .78rem; font-family: monospace;}
.reco {background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; border-radius: 12px;
       padding: 12px 16px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x if x is not None else ""))


def chip(text, color):
    return f'<span class="chip" style="background:{color}">{esc(text)}</span>'


def kpi(value, label, color="#111827"):
    return f'<div class="kpi"><div class="v" style="color:{color}">{esc(value)}</div><div class="l">{esc(label)}</div></div>'

# ---------------------------------------------------------------- accès aux données
if not DEMO:
    from databricks import sql
    from databricks.sdk.core import Config

    _cfg = Config()

    @st.cache_resource
    def _conn():
        return sql.connect(
            server_hostname=_cfg.host.replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
            credentials_provider=lambda: _cfg.authenticate,
        )

    @st.cache_data(ttl=30, show_spinner=False)
    def run_query(q: str) -> pd.DataFrame:
        with _conn().cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
        return pd.DataFrame([tuple(r) for r in rows], columns=cols)


def _demo_frames():
    def row(case, pred, sec, conf, rep, outs):
        return {"batch_id": "DEMO_BATCH", "config": "multi_rag_demo", "run": 1, "case_id": case,
                "predicted": pred, "secondary": json.dumps(sec), "confidence": conf, "n_tools": 5,
                "tools": "get_forecasts,compare_features,compare_configs,check_freshness,get_pipeline_logs",
                "n_llm": 11, "seconds": 16.0, "repaired": False, "raw": json.dumps(rep),
                "tool_outputs": json.dumps(outs), "ts": "2026-01-01T10:00:00"}

    def verif(ok=True):
        return {"passed": ok, "problems": [], "revised": False, "sources_allowed": []}

    fc = lambda b, r: ["get_forecasts", json.dumps({"batch_forecast": b, "realtime_forecast": r,
                                                    "difference": r - b, "difference_pct": round(100 * (r - b) / b, 2)})]
    fr = lambda a, b: ["check_freshness", json.dumps({"batch_age_hours": a, "realtime_age_hours": b})]
    runs = pd.DataFrame([
        row("DEMO_01", "STALE_DATA", [], "high",
            {"primary_cause": "STALE_DATA", "secondary_causes": [], "confidence": "high",
             "evidence": [{"claim": "Le snapshot Batch a 50 h (exemple fictif)", "source": "tool:check_freshness"},
                          {"claim": "Règle : au-delà de 24 h un snapshot est obsolète", "source": "doc:exemple"}],
             "recommendation": "Relancer l'ingestion en amont, puis le Batch (exemple fictif).",
             "verification": verif()}, [fc(950, 1300), fr(50, 1)]),
        row("DEMO_02", "NO_ISSUE", [], "high",
            {"primary_cause": "NO_ISSUE", "secondary_causes": [], "confidence": "high",
             "evidence": [{"claim": "Écart de 3,7 % (exemple fictif)", "source": "tool:get_forecasts"}],
             "recommendation": "Aucune action.", "verification": verif()}, [fc(1500, 1555), fr(2, 0.1)]),
        row("DEMO_03", "CONFIG_MISMATCH", ["FEATURE_MISMATCH"], "medium",
            {"primary_cause": "CONFIG_MISMATCH", "secondary_causes": ["FEATURE_MISMATCH"], "confidence": "medium",
             "evidence": [{"claim": "Horizon 7 contre 14 jours (exemple fictif)", "source": "tool:compare_configs"}],
             "recommendation": "Aligner l'horizon, puis vérifier la promotion.", "verification": verif()},
            [fc(1100, 1600), fr(2, 2)]),
    ])
    batches = pd.DataFrame([{"batch_id": "DEMO_BATCH", "ts": "2026-01-01T10:00:00", "runs": 1}])
    ev = pd.DataFrame([{"config": "3 agents", "batch_id": "DEMO_BATCH", "run": 1, "accuracy_pct": 100.0,
                        "llm_calls": 11.0, "seconds": 16.0}])
    return runs, batches, ev


def _safe_id(x: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(x)):
        raise ValueError("identifiant invalide")
    return str(x)


def load_batches():
    if DEMO:
        return _demo_frames()[1]
    return run_query(f"SELECT batch_id, MIN(ts) AS ts, MAX(run) AS runs FROM {SCHEMA}.agent_runs_v2 "
                     "WHERE config LIKE 'multi%' GROUP BY batch_id ORDER BY ts DESC")


def load_runs(batch_id, run):
    if DEMO:
        return _demo_frames()[0]
    return run_query(f"SELECT * FROM {SCHEMA}.agent_runs_v2 WHERE batch_id = '{_safe_id(batch_id)}' "
                     f"AND run = {int(run)} ORDER BY case_id")


def load_eval():
    if DEMO:
        return _demo_frames()[2]
    return run_query(f"""
        SELECT CASE WHEN a.config LIKE 'multi%' THEN '3 agents' ELSE 'Agent unique' END AS config,
               a.batch_id, a.run,
               ROUND(100.0 * SUM(CASE WHEN a.predicted = k.expected_root_cause THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
               ROUND(AVG(a.n_llm), 1) AS llm_calls, ROUND(AVG(a.seconds), 1) AS seconds
        FROM {SCHEMA}.agent_runs_v2 a JOIN {SCHEMA}.known_issues k ON a.case_id = k.case_id
        GROUP BY 1, a.batch_id, a.run ORDER BY a.batch_id, a.run""")


# ---------------------------------------------------------------- lecture des résultats
def parse_report(raw):
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def tool_data(tool_outputs, name):
    try:
        for item in json.loads(tool_outputs):
            if isinstance(item, list) and len(item) == 2 and item[0] == name:
                return json.loads(item[1])
    except Exception:
        return None
    return None


def parse_list(s):
    try:
        v = json.loads(s or "[]")
        return v if isinstance(v, list) else []
    except Exception:
        return []


def build_cases(df):
    out = []
    for _, r in df.iterrows():
        rep = parse_report(r["raw"]) or {}
        ver = rep.get("verification") or {}
        conf = str(r["confidence"]).lower()
        out.append({
            "Cas": r["case_id"], "Cause principale": r["predicted"],
            "Autres causes": ", ".join(parse_list(r["secondary"])) or "-",
            "Confiance": CONF_ICON.get(conf, "n/d"),
            "Vérification": ("✅ réussie" if ver.get("passed") else "⚠️ à revoir") if ver else "n/d",
            "Recommandation": rep.get("recommendation") or "n/d",
            "_issue": r["predicted"] != "NO_ISSUE", "_conf": CONF_RANK.get(conf, 1),
        })
    cases = pd.DataFrame(out)
    return cases.sort_values(["_issue", "_conf"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------- interface
st.markdown(
    '<div class="hero"><h1>🔎 Forecast Debugging Agent</h1>'
    "<p>Pourquoi une prévision Batch et une prévision Real-Time diffèrent-elles ? "
    "Trois agents enquêtent, citent leurs preuves et proposent une action.</p></div>"
    '<div class="notice">Prototype non commercial · données <b>synthétiques</b>, 12 cas · '
    "l'agent peut se tromper : vérifiez la confiance et la vérification avant d'agir.</div>",
    unsafe_allow_html=True)
if DEMO:
    st.info("Mode démonstration : les cas affichés sont **fictifs**. "
            "Sur Databricks, l'application lit la table agent_runs_v2.")

with st.sidebar:
    st.header("Exécution affichée")
    batches = load_batches()
    if batches.empty:
        st.error("Aucune exécution des 3 agents dans agent_runs_v2. Lancez le notebook 08_multi_agent.")
        st.stop()
    labels = {b: f"{str(t)[:16].replace('T', ' ')} · {int(n)} passe(s)"
              for b, t, n in zip(batches["batch_id"], batches["ts"], batches["runs"])}
    batch_id = st.selectbox("Exécution", list(labels), format_func=lambda b: labels[b])
    n_runs = int(batches.loc[batches["batch_id"] == batch_id, "runs"].iloc[0])
    run = st.selectbox("Passe", list(range(1, n_runs + 1))) if n_runs > 1 else 1
    if st.button("Actualiser"):
        if not DEMO:
            st.cache_data.clear()
        st.rerun()
    st.caption(f"Page lue à {datetime.now().strftime('%H:%M:%S')} (cache de 30 s).")

runs_df = load_runs(batch_id, run)
if runs_df.empty:
    st.error("Cette exécution ne contient aucun cas.")
    st.stop()
cases = build_cases(runs_df)

tab1, tab2, tab3, tab4 = st.tabs(["À traiter", "Détail d'un cas", "Comment ça marche", "Évaluation et limites"])

# ---- 1. triage
with tab1:
    n = len(cases)
    n_issue = int(cases["_issue"].sum())
    n_low = int((cases["Confiance"] == CONF_ICON["low"]).sum())
    n_ver = int((cases["Vérification"] == "⚠️ à revoir").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(n, "Cas analysés"), unsafe_allow_html=True)
    c2.markdown(kpi(n_issue, "Avec un problème", "#B45309"), unsafe_allow_html=True)
    c3.markdown(kpi(n_low, "Confiance faible", "#B91C1C"), unsafe_allow_html=True)
    c4.markdown(kpi(n_ver, "Vérification à revoir", "#B91C1C"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Causes trouvées")
        st.bar_chart(cases["Cause principale"].value_counts())
    with right:
        st.subheader("Liste de triage")
        st.caption("Problèmes d'abord, puis confiance la plus faible : ce sont les cas à relire en priorité.")
        for _, c in cases.iterrows():
            col = CAUSE_COLOR.get(c["Cause principale"], "#9CA3AF")
            reco = str(c["Recommandation"])
            reco = reco if len(reco) <= 140 else reco[:137] + "..."
            st.markdown(
                f'<div class="case" style="border-left-color:{col}"><div class="t">{esc(c["Cas"])} '
                f'{chip(c["Cause principale"], col)}</div>'
                f'<div>Confiance : {esc(c["Confiance"])} &nbsp;·&nbsp; Vérification : {esc(c["Vérification"])}'
                f'{"" if c["Autres causes"] == "-" else " &nbsp;·&nbsp; Autres : " + esc(c["Autres causes"])}</div>'
                f'<div class="r">{esc(reco)}</div></div>', unsafe_allow_html=True)
        with st.expander("Voir sous forme de tableau"):
            st.dataframe(cases.drop(columns=["_issue", "_conf"]), hide_index=True)

# ---- 2. détail
with tab2:
    case = st.selectbox("Cas", cases["Cas"].tolist())
    row = runs_df[runs_df["case_id"] == case].iloc[0]
    rep = parse_report(row["raw"])
    cause = row["predicted"]
    conf = CONF_ICON.get(str(row["confidence"]).lower(), "n/d")
    ccol = CAUSE_COLOR.get(cause, "#9CA3AF")
    cflag = str(row["confidence"]).lower()
    st.markdown(f'<h3 style="margin-bottom:4px">{esc(case)} &nbsp;{chip(cause, ccol)}'
                f'{chip("confiance " + {"high": "haute", "medium": "moyenne", "low": "faible"}.get(cflag, "n/d"), CONF_COLOR.get(cflag, "#9CA3AF"))}</h3>',
                unsafe_allow_html=True)
    st.write(CAUSE_TEXT.get(cause, ""))
    sec = parse_list(row["secondary"])
    if sec:
        st.markdown("Autres causes possibles : " + " ".join(chip(x, CAUSE_COLOR.get(x, "#9CA3AF")) for x in sec),
                    unsafe_allow_html=True)

    fc = tool_data(row["tool_outputs"], "get_forecasts")
    fr = tool_data(row["tool_outputs"], "check_freshness")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Prévisions**")
        if fc:
            st.bar_chart(pd.DataFrame({"Prévision": [fc["batch_forecast"], fc["realtime_forecast"]]},
                                      index=["Batch", "Real-Time"]))
            st.caption(f"Écart : {fc.get('difference')} unités ({fc.get('difference_pct')} %)")
        else:
            st.caption("Données non disponibles (sortie des outils absente ou tronquée).")
    with col_b:
        st.markdown("**Fraîcheur des données**")
        if fr:
            age = float(fr["batch_age_hours"])
            st.metric("Âge du snapshot Batch", f"{age:.0f} h",
                      delta=f"{age - STALE_HOURS:+.0f} h par rapport à la limite de {STALE_HOURS} h",
                      delta_color="inverse")
            st.metric("Âge du snapshot Real-Time", f"{float(fr['realtime_age_hours']):.1f} h")
        else:
            st.caption("Données non disponibles.")

    if rep:
        if rep.get("recommendation"):
            st.markdown(f'<div class="reco"><b>Recommandation</b><br>{esc(rep["recommendation"])}</div>',
                        unsafe_allow_html=True)
        st.markdown("**Preuves et sources**")
        for e in rep.get("evidence", []):
            src = e.get("source") or "source non citée"
            st.markdown(f'<div class="ev">{esc(e.get("claim"))}<div class="s">{esc(src)}</div></div>',
                        unsafe_allow_html=True)
        ver = rep.get("verification") or {}
        with st.expander("Vérification par le code"):
            if ver:
                st.write("Réussie : " + ("oui" if ver.get("passed") else "non")
                         + " · Révisée : " + ("oui" if ver.get("revised") else "non"))
                for p in ver.get("problems", []):
                    st.write("- " + str(p))
                st.caption("Contrôle uniquement les sources citées et les nombres cités, "
                           "pas la logique du raisonnement.")
            else:
                st.write("Aucune information de vérification.")
    else:
        st.info("Le rapport détaillé est illisible ou tronqué ; seules la cause et la confiance sont affichées.")
    st.caption("Outils appelés : " + (row["tools"] or "aucun").replace(",", ", "))

# ---- 3. architecture
with tab3:
    st.subheader("Trois agents, un rapport vérifié")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#4F46E5", fontname="Helvetica"];
      Q [label="Écart Batch ≠ Real-Time\\n(un case_id)", fillcolor="#FEF3C7", color="#F59E0B"];
      A1 [label="Agent 1 · Investigateur\\n5 outils en lecture seule"];
      A2 [label="Agent 2 · Connaissance\\nRAG hybride BM25 + embeddings"];
      A3 [label="Agent 3 · Analyste\\ncause racine + rapport"];
      V  [label="Vérification par le code\\nsources et nombres", fillcolor="#D1FAE5", color="#10B981"];
      R  [label="Rapport\\ncause · preuves · confiance\\nrecommandation", fillcolor="#FEF3C7", color="#F59E0B"];
      Q -> A1; Q -> A2; A1 -> A3 [label="faits"]; A2 -> A3 [label="règles"];
      A3 -> V; V -> A3 [label="1 révision", style=dashed]; V -> R;
    }""")
    st.markdown("""
- **Les outils donnent des faits, jamais un verdict.** Les règles (24 h, 5 %) viennent des documents, retrouvés par le RAG.
- **Lecture seule.** Liste blanche de tables : la vérité terrain n'est pas accessible aux agents.
- **Chaque preuve cite sa source** (`tool:...` ou `doc:...`), et le code vérifie que la source existe et que les nombres cités figurent dans les entrées.
- **Une seule révision**, signalée dans le rapport. Rien n'est caché.
""")

# ---- 4. évaluation
with tab4:
    st.subheader("Résultats mesurés")
    try:
        ev = load_eval()
        if ev.empty:
            st.info("Aucun résultat à comparer avec la vérité terrain.")
        else:
            st.dataframe(ev.rename(columns={"config": "Système", "batch_id": "Exécution", "run": "Passe",
                                            "accuracy_pct": "Exactitude (%)", "llm_calls": "Appels LLM / cas",
                                            "seconds": "Secondes / cas"}), hide_index=True)
            st.bar_chart(ev.assign(Exécution=ev["config"] + " · passe " + ev["run"].astype(str))
                         .set_index("Exécution")["accuracy_pct"])
    except Exception as e:  # droits manquants sur known_issues, par exemple
        st.info(f"Évaluation indisponible : {type(e).__name__}. "
                "Donnez à l'application le droit SELECT sur known_issues.")
    st.subheader("Limites")
    st.markdown("""
- Données synthétiques et 12 cas : une seule erreur change le score de 8 points. Les écarts entre systèmes ne sont pas significatifs.
- Les règles écrites à la main font 12/12 sur ces cas parce qu'elles ont été écrites en les connaissant.
- L'agent n'est pas parfaitement reproductible, même à température 0.
- La vérification ne contrôle pas la logique du raisonnement.
- Les 3 agents coûtent environ deux fois plus d'appels et de temps que l'agent unique.
""")

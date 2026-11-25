"""Thème et composants visuels du Forecast Debugging Agent (à placer à côté de app.py).

Les noms de classes CSS sont ceux de l'ancien app.py (hero, card, kpi, chip, verdict, ev, reco, chain, node...):
le nouveau style s'applique donc tout seul, sans toucher au reste du code.
"""
import html
import re

import streamlit as st

INK, AMBER, INDIGO, TEAL, VIOLET = "#0F1B2D", "#D98E04", "#3B4BD8", "#0E7C86", "#7C3AED"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {font-family: 'Manrope', 'Segoe UI', system-ui, sans-serif;}
.stApp {background: #F3F5F8;}
.block-container {padding-top: 1.2rem; max-width: 1280px;}
h5 {font-weight: 800 !important; color: #0F1B2D; letter-spacing: -.01em; margin-top: .6rem !important;}
[data-testid="stSidebar"] {background: #E8ECF3; border-right: 1px solid #D5DBE6;}

/* En-tête : deux pipelines qui divergent */
.hero {position: relative; overflow: hidden; background: #0F1B2D; color: #fff; border-radius: 20px;
       padding: 30px 36px 28px; margin-bottom: 12px;
       background-image: linear-gradient(90deg, rgba(217,142,4,.22) 0%, rgba(217,142,4,0) 38%),
                         linear-gradient(270deg, rgba(59,75,216,.38) 0%, rgba(59,75,216,0) 42%),
                         repeating-linear-gradient(0deg, rgba(255,255,255,.035) 0 1px, transparent 1px 28px);}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: #fff !important;}
.hero p {margin: 8px 0 0 0; max-width: 680px; font-size: 1.02rem; line-height: 1.5; color: #CBD5E7;}
.hero .pipes {display: flex; align-items: center; gap: 10px; margin-top: 18px; font-size: .84rem; font-weight: 700;}
.hero .pipe {padding: 5px 14px; border-radius: 999px; color: #fff;}
.hero .pipe.b {background: #D98E04;} .hero .pipe.r {background: #3B4BD8;}
.hero .gap {flex: 0 0 90px; height: 0; border-top: 2px dashed #8FA0BE;}
.hero .q {color: #FBBF24; font-size: 1.1rem;}
.notice {background: #FFF4DB; border: 1px solid #F3D48A; color: #6B4600 !important; border-radius: 12px;
         padding: 9px 16px; font-size: .86rem; margin-bottom: 14px;}

/* Cartes */
.card, .kpi, .verdict, .ev, .panel {background: #fff; color: #0F1B2D; border: 1px solid #DDE3EC; border-radius: 14px;
       box-shadow: 0 1px 2px rgba(15,27,45,.05);}
.card {padding: 18px 20px; height: 100%; border-top: 4px solid #3B4BD8;}
.card h4 {margin: 0 0 6px 0; font-size: 1.02rem; font-weight: 800; color: #0F1B2D;}
.card p {margin: 0; color: #4B5A72; font-size: .9rem; line-height: 1.5;}
.kpi {padding: 14px 18px;}
.kpi .v {font-size: 2rem; font-weight: 800; line-height: 1.05; letter-spacing: -.02em;}
.kpi .l {color: #5B6B84; font-size: .82rem; margin-top: 4px;}
.chip {display: inline-block; padding: 3px 11px; border-radius: 8px; font-size: .78rem; font-weight: 700;
       color: #fff !important; margin-right: 6px;}
.verdict {border-left: 9px solid #9CA3AF; padding: 18px 22px; margin: 6px 0 14px 0;}
.verdict h2 {margin: 0 0 8px 0; font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; color: #0F1B2D;}
.verdict p {margin: 4px 0 0 0; color: #33425B; font-size: .98rem;}
.reco {background: #E6F6F1; border: 1px solid #9ADBC6; border-left: 6px solid #0E9F78; color: #064E3B;
       border-radius: 12px; padding: 14px 18px; margin: 10px 0; line-height: 1.55;}
.empty {text-align: center; color: #5B6B84; padding: 22px 0 10px 0;}

.reco p {margin: 8px 0 0 0;} .reco ol {margin: 8px 0 0 0; padding-left: 22px;} .reco li {margin: 5px 0;}

/* Chaîne d'explication */
.chain {display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 6px 0 14px 0;}
.node {background: #fff; color: #0F1B2D; border: 1px solid #CBD3F5; border-top: 4px solid #3B4BD8;
       border-radius: 10px; padding: 8px 14px; font-size: .88rem;}
.arrow {color: #8A97AD; font-size: 1.3rem; align-self: center;}

/* Comparaison Batch / Real-Time */
.pipe-card {padding: 16px 20px;}
.pipe-card.b {border-top: 5px solid #D98E04;} .pipe-card.r {border-top: 5px solid #3B4BD8;}
.pipe-card h4 {margin: 0 0 10px 0; font-size: .95rem; font-weight: 800;}
.pipe-card .big {font-size: 2.3rem; font-weight: 800; letter-spacing: -.03em; line-height: 1;}
.pipe-card .big small {font-size: .85rem; font-weight: 500; color: #5B6B84; margin-left: 6px;}
.pipe-card dl {display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 14px 0 0 0; font-size: .86rem;}
.pipe-card dt {color: #5B6B84;} .pipe-card dd {margin: 0; font-weight: 700; text-align: right;}
.meter {margin: 10px 0 4px 0;}
.meter .track {position: relative; height: 14px; border-radius: 999px; background: #E3E8F1; overflow: visible;}
.meter .fill {height: 100%; border-radius: 999px;}
.meter .tick {position: absolute; top: -5px; width: 2px; height: 24px; background: #0F1B2D;}
.meter .lab {display: flex; justify-content: space-between; font-size: .78rem; color: #5B6B84; margin-top: 8px;}

/* Preuves */
.ev {padding: 12px 16px; margin-bottom: 8px; line-height: 1.5; font-size: .93rem;}
.ev .s {margin-top: 7px; display: flex; align-items: center; gap: 8px;}
.ev .src {font-family: 'JetBrains Mono', monospace; font-size: .76rem; color: #4B5A72;}
.badge {display: inline-block; padding: 1px 9px; border-radius: 6px; font-size: .72rem; font-weight: 800; color: #fff;}

/* Déroulé de l'enquête */
.trace {position: relative; margin: 4px 0 8px 12px; padding-left: 26px; border-left: 3px solid #CBD3F5;}
.trace .st {position: relative; margin-bottom: 16px;}
.trace .st::before {content: ""; position: absolute; left: -36px; top: 3px; width: 13px; height: 13px;
                    border-radius: 50%; background: #fff; border: 3px solid var(--c, #3B4BD8);}
.trace .t {font-weight: 800; font-size: .94rem;}
.trace .d {color: #4B5A72; font-size: .88rem; margin-top: 2px; line-height: 1.5;}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def e(x):
    return html.escape(str(x if x is not None else ""))


def hero(title, subtitle):
    return (f'<div class="hero"><h1>{e(title)}</h1><p>{e(subtitle)}</p>'
            '<div class="pipes"><span class="pipe b">Batch</span><span class="gap"></span><span class="q">?</span>'
            '<span class="gap"></span><span class="pipe r">Real-Time</span></div></div>')


def meter(value, limit, unit, color):
    """Barre de valeur avec un repère sur la limite documentée."""
    top = max(float(limit) * 2, float(value) * 1.1, 1.0)
    w, t = min(100.0, 100 * float(value) / top), 100 * float(limit) / top
    return (f'<div class="meter"><div class="track"><div class="fill" style="width:{w:.1f}%;background:{color}"></div>'
            f'<div class="tick" style="left:{t:.1f}%"></div></div>'
            f'<div class="lab"><span>{float(value):g} {e(unit)}</span><span>limite documentée : {float(limit):g} {e(unit)}</span></div></div>')


def compare_panel(f, stale_h, tol_pct):
    """Deux cartes côte à côte : Batch et Real-Time (prévision, âge du snapshot, version du modèle, statut)."""
    fc, fr, logs = f["fc"], f["fr"], f["logs"]
    model = next((c for c in f["cfgs"] if c.get("name") == "model_version"), None)

    def card(cls, name, fval, age, mval, side):
        status = (logs.get(side) or {}).get("status", "n/d")
        rows = (f"<dt>Âge du snapshot</dt><dd>{float(age):g} h</dd>" if age is not None else "") + \
               (f"<dt>Version du modèle</dt><dd>{e(mval)}</dd>" if mval is not None else "") + \
               f"<dt>Statut du pipeline</dt><dd>{e(status)}</dd>"
        big = f'{float(fval):g}<small>unités</small>' if fval is not None else "n/d"
        return f'<div class="panel pipe-card {cls}"><h4>{name}</h4><div class="big">{big}</div><dl>{rows}</dl></div>'

    b = card("b", "Pipeline Batch", (fc or {}).get("batch_forecast"), (fr or {}).get("batch_age_hours"),
             (model or {}).get("batch_value"), "batch")
    r = card("r", "Pipeline Real-Time", (fc or {}).get("realtime_forecast"), (fr or {}).get("realtime_age_hours"),
             (model or {}).get("realtime_value"), "realtime")
    m = ""
    if fc:
        pct = abs(float(fc["difference_pct"]))
        m += ('<div class="panel" style="padding:14px 20px;margin-top:10px"><b>Écart mesuré</b>'
              + meter(pct, tol_pct, "%", "#B45309" if pct > tol_pct else "#0E9F78") + "</div>")
    if fr:
        age = float(fr["batch_age_hours"])
        m += ('<div class="panel" style="padding:14px 20px;margin-top:10px"><b>Fraîcheur du snapshot Batch</b>'
              + meter(age, stale_h, "h", "#B91C1C" if age > stale_h else "#0E9F78") + "</div>")
    return b, r, m


def evidence_html(ev):
    src = str(ev.get("source") or "")
    kind, _, name = src.partition(":")
    label, color = {"tool": ("Outil", TEAL), "doc": ("Document", VIOLET)}.get(kind, ("Non citée", "#9CA3AF"))
    shown = name if name else "source non citée"
    return (f'<div class="ev">{e(ev.get("claim"))}<div class="s"><span class="badge" style="background:{color}">{label}</span>'
            f'<span class="src">{e(shown)}</span></div></div>')


def trace_html(f, conf_fr, ver):
    steps = [
        (TEAL, f"Agent 1 · Investigateur : {len(f['tools'])} outils interrogés", ", ".join(f["tools"]) or "aucun outil"),
        (VIOLET, f"Agent 2 · Connaissance : {len(f['docs'])} document(s) retrouvé(s)", ", ".join(f["docs"]) or "aucun document"),
        (INDIGO, f"Agent 3 · Analyste : cause retenue {f['cause']}",
         f"confiance {conf_fr}" + (f" · autres causes : {', '.join(f['secondary'])}" if f["secondary"] else "")),
        ("#0E9F78", f"Vérification par le code : {ver}", "contrôle des sources citées et des nombres cités"),
    ]
    body = "".join(f'<div class="st" style="--c:{c}"><div class="t">{e(t)}</div><div class="d">{e(d)}</div></div>'
                   for c, t, d in steps)
    return f'<div class="trace">{body}</div>'


def split_reco(text):
    """Sépare « 1. ... 2. ... » en (introduction, [étapes]). Sans numérotation : (texte, [])."""
    s = str(text or "").strip()
    marks, n = [], 1
    for m in re.finditer(r"(?:(?<=\s)|^)(\d{1,2})[.)]\s+", s):
        if int(m.group(1)) == n:      # accepte seulement 1, 2, 3... dans l'ordre
            marks.append((m.start(), m.end()))
            n += 1
    if len(marks) < 2:
        return s, []
    intro = s[:marks[0][0]].strip()
    items = [s[b:(marks[i + 1][0] if i + 1 < len(marks) else len(s))].strip() for i, (a, b) in enumerate(marks)]
    return intro, [x for x in items if x]


def reco_html(text):
    intro, items = split_reco(text)
    out = '<div class="reco"><b>Recommandation</b>'
    if intro:
        out += f"<p>{e(intro)}</p>"
    if items:
        out += "<ol>" + "".join(f"<li>{e(i)}</li>" for i in items) + "</ol>"
    return out + "</div>"

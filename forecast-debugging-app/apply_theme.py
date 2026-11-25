"""Applique le thème (theme.py) à app.py : `python apply_theme.py app.py`
Crée une sauvegarde app.py.bak, fait les 4 remplacements et s'arrête si un repère est introuvable.
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "app.py")
src = path.read_text(encoding="utf-8")
if "import theme" in src:
    sys.exit("app.py utilise déjà theme : rien à faire.")

HERO = '''st.markdown(theme.hero("Forecast Debugging Agent",
    "Pourquoi une prévision Batch et une prévision Real-Time diffèrent-elles ? "
    "Trois agents enquêtent, citent leurs preuves et proposent une action."), unsafe_allow_html=True)
st.markdown('<div class="notice">Prototype non commercial · données <b>synthétiques</b> · '
            "l'agent peut se tromper : vérifiez la confiance et la vérification avant d'agir.</div>",
            unsafe_allow_html=True)'''

COMPARE = '''            b_html, r_html, m_html = theme.compare_panel(f, STALE_HOURS, TOLERANCE_PCT)
            cb, cr = st.columns(2)
            cb.markdown(b_html, unsafe_allow_html=True)
            cr.markdown(r_html, unsafe_allow_html=True)
            st.markdown(m_html, unsafe_allow_html=True)
            with st.expander("Déroulé de l'enquête", expanded=True):
                st.markdown(theme.trace_html(f, CONF_FR.get(f["conf"], "n/d"), ver_label(rep)), unsafe_allow_html=True)
'''

EVID = '''for ev in rep["evidence"]:
                    st.markdown(theme.evidence_html(ev), unsafe_allow_html=True)'''

steps = [
    # (description, motif, remplacement, drapeaux)
    ("bloc CSS", r'st\.markdown\("""\s*<style>.*?</style>\s*""", unsafe_allow_html=True\)',
     "import theme\ntheme.inject()", re.S),
    ("en-tête", r"st\.markdown\(\s*'<div class=\"hero\">.*?unsafe_allow_html=True\)", HERO, re.S),
    ("compteurs", r'(k\[3\]\.markdown\(kpi\(.*?"Âge du snapshot Real-Time"\), unsafe_allow_html=True\)\n\s*st\.write\(""\)\n)',
     lambda m: m.group(1) + COMPARE, re.S),
    ("preuves", r'for e in rep\["evidence"\]:\n\s+st\.markdown\(f\'<div class="ev">.*?unsafe_allow_html=True\)',
     EVID, re.S),
]
for name, pattern, repl, flags in steps:
    src, n = re.subn(pattern, repl, src, count=1, flags=flags)
    if n != 1:
        sys.exit(f"Repère introuvable ({name}) : app.py a été modifié par rapport à la version attendue. Rien n'a été écrit.")

path.with_suffix(".py.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
path.write_text(src, encoding="utf-8")
print("OK : theme appliqué. Sauvegarde :", path.with_suffix(".py.bak"))

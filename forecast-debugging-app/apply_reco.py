"""Affiche les recommandations en liste numérotée (page + PDF du cas) et en entier (PDF du lot).
Usage : python apply_reco.py   (dans le dossier qui contient app.py et theme.py, déjà passés par apply_theme.py)
Sauvegardes : app.py.reco.bak et theme.py.reco.bak. Rien n'est écrit si un repère est introuvable.
"""
import re
import sys
from pathlib import Path

app_p, theme_p = Path("app.py"), Path("theme.py")
app, theme = app_p.read_text(encoding="utf-8"), theme_p.read_text(encoding="utf-8")
if "reco_html" in theme or "split_reco" in app:
    sys.exit("Déjà appliqué : rien à faire.")
if "import theme" not in app:
    sys.exit("Lancez d'abord apply_theme.py : app.py n'utilise pas encore theme.")

# ---------------------------------------------------------------- theme.py
CSS_ADD = """.reco p {margin: 8px 0 0 0;} .reco ol {margin: 8px 0 0 0; padding-left: 22px;} .reco li {margin: 5px 0;}
"""
FUNCS = '''

def split_reco(text):
    """Sépare « 1. ... 2. ... » en (introduction, [étapes]). Sans numérotation : (texte, [])."""
    s = str(text or "").strip()
    marks, n = [], 1
    for m in re.finditer(r"(?:(?<=\\s)|^)(\\d{1,2})[.)]\\s+", s):
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
'''
if "/* Chaîne d'explication */" not in theme or "import html\n" not in theme:
    sys.exit("Repère introuvable dans theme.py. Rien n'a été écrit.")
theme = theme.replace("import html\n", "import html\nimport re\n", 1)
theme = theme.replace("/* Chaîne d'explication */", CSS_ADD + "\n/* Chaîne d'explication */", 1)
theme = theme.rstrip("\n") + "\n" + FUNCS

# ---------------------------------------------------------------- app.py
UI = 'st.markdown(theme.reco_html(rep["recommendation"]), unsafe_allow_html=True)'
CASE_PDF = '''intro, steps = theme.split_reco(rep["recommendation"])
        story += [P("Recommandation", "h2")]
        if intro:
            story.append(P(intro))
        story += [P(f"{i}. {t}") for i, t in enumerate(steps, 1)]'''
BATCH_PDF = '''    story += [P("Detail par cas", "h2")]
    data = [[P(h) for h in ["Cas", "Cause", "Conf.", "Verif.", "Ecart %", "Age Batch h"]]]
    for _, r in table.iterrows():
        data.append([P(r["Cas"]), P(r["Cause"]), P(r["Confiance"]), P(r["Vérification"]),
                     P(r["Écart (%)"]), P(r["Âge snapshot Batch (h)"])])
    story.append(_pdf_table(data, [2.4 * cm, 5.2 * cm, 1.8 * cm, 2.0 * cm, 1.9 * cm, 2.4 * cm]))
    story += [P("Recommandations", "h2")]
    for _, r in table.iterrows():
        intro, steps = theme.split_reco(r["Recommandation"])
        story.append(P(f"{r['Cas']} - {r['Cause']}", "body"))
        if intro:
            story.append(P(intro))
        story += [P(f"{i}. {t}") for i, t in enumerate(steps, 1)]
        story.append(Spacer(1, 6))'''
steps = [
    ("recommandation de la page", r"st\.markdown\(f'<div class=\"reco\">.*?unsafe_allow_html=True\)", UI),
    ("recommandation du PDF d'un cas", r'story \+= \[P\("Recommandation", "h2"\), P\(rep\["recommendation"\]\)\]', CASE_PDF),
    ("tableau du PDF du lot",
     r'    story \+= \[P\("Detail par cas", "h2"\)\].*?\[1\.9 \* cm, 3\.4 \* cm, 1\.5 \* cm, 1\.6 \* cm, 1\.5 \* cm, 1\.7 \* cm, 5\.6 \* cm\]\)\)',
     BATCH_PDF),
]
for name, pattern, repl in steps:
    app, n = re.subn(pattern, lambda m, r=repl: r, app, count=1, flags=re.S)
    if n != 1:
        sys.exit(f"Repère introuvable ({name}) dans app.py. Rien n'a été écrit.")

Path("app.py.reco.bak").write_text(app_p.read_text(encoding="utf-8"), encoding="utf-8")
Path("theme.py.reco.bak").write_text(theme_p.read_text(encoding="utf-8"), encoding="utf-8")
app_p.write_text(app, encoding="utf-8")
theme_p.write_text(theme, encoding="utf-8")
print("OK : recommandations en liste numérotée (page, PDF du cas) et en entier (PDF du lot).")

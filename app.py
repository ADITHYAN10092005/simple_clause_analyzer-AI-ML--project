"""
Streamlit Web UI for Legal Document Clause Risk Analyzer
KTU B.Tech AI & ML Course Mini Project

Only the visual layer lives here. Model training, document parsing, clause analysis
and the risk calculation are identical to the original app.
"""
import os
import re
import html
import math
from datetime import datetime

import pandas as pd
import pypdf
import docx
import streamlit as st
try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Core ML & rule-agent logic, imported unchanged
from simple_clause_analyzer import (
    load_cuad,
    preprocess,
    extract_facts,
    run_rule_agent,
    split_clauses,
    STANDARD_CLAUSES,
    TARGET_CATEGORIES,
    BASE_RISKS,
    DEMO_CONTRACT
)

st.set_page_config(
    page_title="ClauseInspect",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# THEME
# =====================================================================
NAV_ITEMS = ["Dashboard", "Clause Details", "Classifier Metrics", "K-Means Clustering"]

# Material icon names shown next to each navigation button
NAV_ICONS = [":material/dashboard:", ":material/description:", ":material/bar_chart:", ":material/scatter_plot:"]

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root { --font: 'Inter', 'Segoe UI', system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial, sans-serif; }

/* one typeface everywhere (Material icon glyphs are left alone) */
html, body, .stApp,
.stApp *:not([data-testid='stIconMaterial']):not(.material-icons) {
    font-family: var(--font) !important;
}
.stApp { background: #eceef1; -webkit-font-smoothing: antialiased; }
[data-testid='stHeader'] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.25rem 1.75rem 3rem; max-width: 1360px; }

[data-testid='stMain'] p, [data-testid='stMain'] label,
[data-testid='stMain'] h2, [data-testid='stMain'] h3,
section.main p, section.main label, section.main h2, section.main h3 { color: #111827; }

/* ---------- Top bar ---------- */
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 16px;
          background: #fff; border: 1px solid #e3e6ea; border-radius: 12px;
          padding: 14px 20px; margin-bottom: 16px; }
.title { font-size: 1.25rem; font-weight: 600; color: #0b0b0d; }
.doc-chip { font-size: 0.8rem; font-weight: 500; color: #4b5563; background: #f3f4f6;
            border-radius: 6px; padding: 6px 12px; max-width: 50%;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---------- Cards ---------- */
.card { background: #fff; border: 1px solid #e3e6ea; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; }
.card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; gap: 8px; }
.card-title { font-size: 1rem; font-weight: 600; color: #0b0b0d; }
.more { display: inline-flex; align-items: center; }
.legend-chip { display: inline-block; border: 1px solid; border-radius: 999px; padding: 2px 10px;
               font-size: 0.7rem; font-weight: 500; margin-left: 6px; }

/* ---------- KPI row ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 16px; }
.kpi { background: #fff; border: 1px solid #e3e6ea; border-radius: 12px; padding: 16px 18px 14px; }
.kpi-label { font-size: 0.8rem; color: #4b5563; font-weight: 500; }
.kpi-value { font-size: 2.2rem; font-weight: 500; color: #0b0b0d; line-height: 1.1; margin: 8px 0 6px; letter-spacing: -0.02em; }
.delta { display: inline-flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 500; }
.delta::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.delta.up { color: #15803d; } .delta.down { color: #b91c1c; } .delta.flat { color: #6b7280; }
@media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }

/* ---------- Table ---------- */
.tbl-wrap { overflow-x: auto; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.tbl th { text-align: left; font-weight: 500; color: #6b7280; font-size: 0.74rem; padding: 8px 8px 10px 0; border-bottom: 1px solid #eceef1; }
.tbl td { padding: 13px 8px 13px 0; border-bottom: 1px solid #f1f2f5; color: #111827; vertical-align: middle; }
.tbl td.name { font-weight: 600; }
.tbl tr:last-child td { border-bottom: none; }
.tbl.num th:not(:first-child), .tbl.num td:not(:first-child) { text-align: right; padding-right: 16px; }
.foot { font-size: 0.72rem; color: #9ca3af; margin-top: 10px; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 500;
        border: 1px solid; white-space: nowrap; }

/* ---------- Right column ---------- */
.file-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.file-ico { width: 32px; height: 38px; border-radius: 5px; font-size: 0.6rem; font-weight: 700;
            display: flex; align-items: flex-end; justify-content: center; padding-bottom: 5px; border: 1px solid; flex-shrink: 0; }
.file-name { font-weight: 600; font-size: 0.88rem; color: #111827; word-break: break-all; }
.file-meta { font-size: 0.72rem; color: #9ca3af; margin-top: 2px; }
.kv { display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; margin: 10px 0; color: #6b7280; gap: 10px; }
.kv b { color: #111827; font-weight: 500; text-align: right; }
.prog-head { display: flex; justify-content: space-between; align-items: baseline; margin-top: 18px; font-size: 0.88rem; font-weight: 600; color: #111827; }
.prog-head span { font-size: 0.72rem; color: #9ca3af; font-weight: 400; }
.prog { height: 5px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 8px 0 12px; }
.prog div { height: 100%; background: linear-gradient(90deg, #2c6e8f, #12a594); border-radius: 999px; }
.stage { display: inline-block; border: 1px solid #e5e7eb; border-radius: 999px; padding: 2px 10px; font-size: 0.7rem; color: #6b7280; }
.reco { background: #eeeefc; border-radius: 8px; padding: 12px 14px; margin: 16px 0 14px; }
.reco-title { font-size: 0.82rem; font-weight: 600; color: #1f2937; margin-bottom: 4px; }
.reco-body { font-size: 0.76rem; color: #4b5563; line-height: 1.55; }
.action-bar { display: flex; justify-content: space-between; align-items: center; gap: 10px; background: #0b0b0d;
              border-radius: 8px; padding: 12px 14px; font-size: 0.8rem; }
.action-bar span { color: #9ca3af; }
.action-bar b { color: #fff; font-weight: 600; }
.mini-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 0; border-bottom: 1px solid #f1f2f5; }
.mini-row:first-of-type { padding-top: 0; }
.mini-row:last-child { border-bottom: none; padding-bottom: 0; }
.mini-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
.mini-ico { width: 30px; height: 34px; border-radius: 5px; background: #f3f4f6; color: #374151; font-weight: 600;
            font-size: 0.7rem; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.mini-name { font-size: 0.84rem; font-weight: 600; color: #111827; }
.mini-meta { font-size: 0.7rem; color: #9ca3af; }

/* ---------- Clause detail cards ---------- */
.clause { background: #fff; border: 1px solid #e3e6ea; border-left: 4px solid var(--rc); border-radius: 12px;
          padding: 18px 20px; margin-bottom: 14px; }
.clause-head { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.clause-title { font-weight: 600; font-size: 0.98rem; color: #111827; }
.clause-title .type { color: #2c6e8f; }
.clause-text { color: #374151; font-size: 0.9rem; line-height: 1.65; background: #f7f8fa; border-radius: 8px;
               padding: 12px 14px; margin-bottom: 14px; }
.reasons-title { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; margin-bottom: 6px; }
.reasons { margin: 0 0 14px 18px; padding: 0; color: #374151; font-size: 0.84rem; line-height: 1.6; }
.sim-row { display: flex; align-items: center; gap: 12px; font-size: 0.78rem; color: #4b5563; flex-wrap: wrap; }
.sim-track { flex: 1; height: 5px; background: #e5e7eb; border-radius: 999px; overflow: hidden; max-width: 240px; min-width: 100px; }
.sim-fill { height: 100%; background: #2c6e8f; border-radius: 999px; }

/* ---------- Main-area inputs (filters): light, not the theme's dark default ---------- */
[data-testid='stMain'] [data-baseweb='select'] > div,
section.main [data-baseweb='select'] > div {
    background: #fff; border: 1px solid #dfe2e7; border-radius: 8px; min-height: 42px;
}
[data-testid='stMain'] [data-baseweb='select'] input,
section.main [data-baseweb='select'] input { color: #111827 !important; }
[data-testid='stMain'] [data-baseweb='select'] svg,
section.main [data-baseweb='select'] svg { fill: #6b7280; }
[data-testid='stMain'] [data-baseweb='tag'],
section.main [data-baseweb='tag'] { background: #eef0f3 !important; border-radius: 6px; }
[data-testid='stMain'] [data-baseweb='tag'] *,
section.main [data-baseweb='tag'] * { color: #111827 !important; }
[data-testid='stMain'] [data-baseweb='tag'] svg,
section.main [data-baseweb='tag'] svg { fill: #6b7280 !important; }
[data-testid='stToolbar'] * { color: #4b5563; }

/* ---------- Sidebar ---------- */
section[data-testid='stSidebar'][aria-expanded='true'] { width: 260px; min-width: 260px; max-width: 260px; }
section[data-testid='stSidebar'] { background: #0b0b0d; border-right: none; }
section[data-testid='stSidebar'] [data-testid='stSidebarUserContent'] { padding: 1.25rem 0.9rem 2rem; }
section[data-testid='stSidebar'] * { color: #d1d5db; }
section[data-testid='stSidebar'] hr { border-color: #26262a; }
section[data-testid='stSidebar'] button { background: #26262a; border: 1px solid #3a3a40; }
section[data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'] { background: #141417; border: 1px dashed #3a3a40; }
section[data-testid='stSidebar'] [data-baseweb='textarea'], section[data-testid='stSidebar'] textarea { background: #141417 !important; border-color: #2b2b30 !important; }
section[data-testid='stSidebar'] [data-baseweb='select'] > div { background: #141417; border: 1px solid #2b2b30; border-radius: 8px; }
section[data-testid='stSidebar'] [data-baseweb='select'] svg { fill: #9ca3af; }

.brand { display: flex; align-items: center; gap: 10px; font-size: 1.1rem; font-weight: 600; color: #fff !important; padding: 2px 6px 6px; }
.brand svg { width: 22px; height: 22px; }
.nav-label { font-size: 0.66rem; letter-spacing: 0.12em; color: #6b7280 !important; margin: 22px 0 8px 6px; text-transform: uppercase; font-weight: 500; }

/* navigation: sidebar buttons styled as a menu (inactive = secondary, active = primary) */
section[data-testid='stSidebar'] button[kind='secondary'],
section[data-testid='stSidebar'] button[kind='primary'] {
    width: 100%; justify-content: flex-start; gap: 12px; padding: 9px 12px; min-height: 0;
    border: none; border-radius: 8px; box-shadow: none !important; text-align: left;
}
section[data-testid='stSidebar'] button[kind='secondary'] { background: transparent; }
section[data-testid='stSidebar'] button[kind='secondary'],
section[data-testid='stSidebar'] button[kind='secondary'] * { color: #9ca3af; }
section[data-testid='stSidebar'] button[kind='secondary']:hover { background: #17171a; }
section[data-testid='stSidebar'] button[kind='primary'],
section[data-testid='stSidebar'] button[kind='primary']:hover,
section[data-testid='stSidebar'] button[kind='primary']:focus,
section[data-testid='stSidebar'] button[kind='primary']:active { background: #fff; border: none; }
section[data-testid='stSidebar'] button[kind='primary'],
section[data-testid='stSidebar'] button[kind='primary'] * { color: #0b0b0d; }
section[data-testid='stSidebar'] button > div { justify-content: flex-start; }
section[data-testid='stSidebar'] button p { font-size: 0.86rem; font-weight: 500; margin: 0; text-align: left; }
section[data-testid='stSidebar'] button[kind='primary'] p { font-weight: 600; }
/* the uploader's own Browse button keeps the regular dark-button look */
section[data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'] button {
    width: auto !important; justify-content: center; padding: 0.25rem 0.75rem;
    background: #26262a !important; border: 1px solid #3a3a40 !important; border-radius: 8px;
}
section[data-testid='stSidebar'] [data-testid='stFileUploaderDropzone'] button * { color: #d1d5db !important; }

.model-row { display: flex; justify-content: space-between; font-size: 0.8rem; padding: 9px 6px; border-bottom: 1px solid #1c1c20; gap: 10px; }
.model-row:last-child { border-bottom: none; }
.model-row b { color: #fff !important; font-weight: 600; }
.side-note { font-size: 0.72rem; color: #6b7280 !important; padding: 0 6px; line-height: 1.5; }
</style>
"""
CUSTOM_CSS = BASE_CSS

# =====================================================================
# CACHED MODEL TRAINING & VECTORIZATION
# =====================================================================
@st.cache_resource(show_spinner="Training models on the CUAD dataset (one-time setup)...")
def get_trained_pipeline(cuad_path="CUADv1.json"):
    df = load_cuad(cuad_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    std_vectors = {cat: vec.transform([preprocess(c)]) for cat, c in STANDARD_CLAUSES.items()}

    # Train 3 models
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_vec, y_train)
    nb = MultinomialNB().fit(X_train_vec, y_train)
    svm = LinearSVC(random_state=42).fit(X_train_vec, y_train)

    accs = {
        'Logistic Regression (Primary)': accuracy_score(y_test, lr.predict(X_test_vec)),
        'Multinomial Naive Bayes': accuracy_score(y_test, nb.predict(X_test_vec)),
        'Linear SVM': accuracy_score(y_test, svm.predict(X_test_vec))
    }

    report = classification_report(y_test, lr.predict(X_test_vec), output_dict=True)
    cm = confusion_matrix(y_test, lr.predict(X_test_vec))

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train_vec)
    crosstab = pd.crosstab(y_train, clusters, rownames=['True Clause Type'], colnames=['Cluster'])

    return {
        'vec': vec,
        'model': lr,
        'std_vectors': std_vectors,
        'accuracies': accs,
        'report': report,
        'confusion_matrix': cm,
        'crosstab': crosstab,
        'dataset_size': len(df),
        'class_names': list(lr.classes_)
    }

# =====================================================================
# DOCUMENT PARSER FOR UPLOADED FILES
# =====================================================================
def extract_text_from_upload(uploaded_file):
    """Parses PDF, DOCX, PPTX, or TXT directly from Streamlit uploaded file buffer."""
    filename = uploaded_file.name.lower()

    if filename.endswith('.pdf'):
        reader = pypdf.PdfReader(uploaded_file)
        return '\n\n'.join(page.extract_text() or '' for page in reader.pages)

    if filename.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        texts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        texts.append(cell.text.strip())
        return '\n\n'.join(texts)

    if filename.endswith('.pptx'):
        if not HAS_PPTX:
            st.error("python-pptx is not installed in this Python environment. Run `pip install python-pptx` to enable PowerPoint parsing.")
            return ''
        prs = Presentation(uploaded_file)
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        if p.text.strip():
                            texts.append(p.text.strip())
                if shape.has_table:
                    for row in shape.table.rows:
                        for cell in row.cells:
                            if cell.text.strip():
                                texts.append(cell.text.strip())
        return '\n\n'.join(texts)

    if filename.endswith('.txt'):
        return uploaded_file.read().decode('utf-8', errors='ignore')

    if filename.endswith('.doc') or filename.endswith('.ppt'):
        st.error(f"{uploaded_file.name} is in a legacy binary format. Open it in Office, Google Docs or Slides and save it as .docx or .pptx.")
        return ''

    st.error(f"Unsupported file format: {uploaded_file.name}")
    return ''

# =====================================================================
# UI HELPERS
# =====================================================================
RISK_COLOR_MAP = {
    'High': '#ef4444',
    'Medium-High': '#f97316',
    'Medium': '#eab308',
    'Low': '#22c55e'
}

ACTION_LABELS = {
    'FLAG-HIGH-RISK': 'Flag high risk',
    'SUGGEST-REWRITE': 'Suggest rewrite',
    'FLAG-CAUTION': 'Flag caution',
    'ACCEPT': 'Accept'
}

# (text, background, border) for the outlined status pills
PILL_STYLES = {
    'High': ('#b91c1c', '#fef2f2', '#fecaca'),
    'Medium-High': ('#c2410c', '#fff7ed', '#fed7aa'),
    'Medium': ('#a16207', '#fefce8', '#fde68a'),
    'Low': ('#15803d', '#f0fdf4', '#bbf7d0'),
}
ACTION_KIND = {
    'FLAG-HIGH-RISK': 'High',
    'SUGGEST-REWRITE': 'Medium-High',
    'FLAG-CAUTION': 'Medium',
    'ACCEPT': 'Low',
}
SEVERITY = {'High': 4, 'Medium-High': 3, 'Medium': 2, 'Low': 1}
RISK_SCORE = {'Low': 25, 'Medium': 50, 'Medium-High': 75, 'High': 100}
FILE_ICON_STYLES = {
    'PDF': ('#fef2f2', '#dc2626', '#fecaca'),
    'DOCX': ('#eff6ff', '#2563eb', '#bfdbfe'),
    'PPTX': ('#fff7ed', '#ea580c', '#fed7aa'),
    'TXT': ('#f3f4f6', '#4b5563', '#e5e7eb'),
}
PAGE_TITLES = {
    'Dashboard': 'AI Analysis Overview',
    'Clause Details': 'Clause Details',
    'Classifier Metrics': 'Classifier Metrics',
    'K-Means Clustering': 'K-Means Clustering',
}

BRAND_MARK = (
    "<svg viewBox='0 0 24 24' fill='none' stroke='#fff' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M12 3l9 4.5-9 4.5-9-4.5z'/><path d='M3 12l9 4.5 9-4.5'/><path d='M3 16.5l9 4.5 9-4.5'/></svg>"
)
MORE = (
    "<span class='more'><svg width='16' height='4' viewBox='0 0 16 4' fill='#9ca3af'>"
    "<circle cx='2' cy='2' r='1.5'/><circle cx='8' cy='2' r='1.5'/><circle cx='14' cy='2' r='1.5'/></svg></span>"
)

def esc(value):
    """Escape text before placing it inside custom HTML."""
    return html.escape(str(value))

def flat(s):
    """Collapse newlines and indentation so Markdown never treats the HTML as a code block."""
    return re.sub(r'\n\s*', '', s.strip())

def pill(label, kind):
    color, bg, border = PILL_STYLES.get(kind, ('#374151', '#f3f4f6', '#e5e7eb'))
    return f"<span class='pill' style='color:{color};background:{bg};border-color:{border};'>{esc(label)}</span>"

def kpi(label, value, delta_text, delta_cls):
    return flat(f"""
    <div class='kpi'>
        <div class='kpi-label'>{esc(label)}</div>
        <div class='kpi-value'>{esc(value)}</div>
        <div class='delta {delta_cls}'>{esc(delta_text)}</div>
    </div>""")

def smooth_path(pts, y_min, y_max):
    """Catmull-Rom spline converted to cubic Bezier segments."""
    if len(pts) < 2:
        return ""
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1x, c1y = p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6
        c2x, c2y = p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6
        c1y = min(max(c1y, y_min), y_max)
        c2y = min(max(c2y, y_min), y_max)
        d += f" C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d

def trend_chart(results):
    """Inline SVG line chart of similarity % and risk score for every clause."""
    W, H = 760, 270
    L, R, T, B = 40, 16, 16, 34
    pw, ph = W - L - R, H - T - B
    n = len(results)
    xs = [L + pw / 2] if n == 1 else [L + i * pw / (n - 1) for i in range(n)]
    ypos = lambda v: T + ph * (1 - v / 100)

    sim_pts = [(xs[i], ypos(min(max(r['sim'], 0), 1) * 100)) for i, r in enumerate(results)]
    risk_pts = [(xs[i], ypos(RISK_SCORE.get(r['risk'], 0))) for i, r in enumerate(results)]

    parts = [f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='width:100%;height:auto;display:block;'>"]

    bw = pw / max(n, 1) * 0.9
    for i, x in enumerate(xs):
        if i % 2 == 0:
            parts.append(f"<rect x='{x - bw / 2:.1f}' y='{T}' width='{bw:.1f}' height='{ph}' fill='#f6f7f9'/>")

    for v in (0, 25, 50, 75, 100):
        y = ypos(v)
        parts.append(f"<line x1='{L}' y1='{y:.1f}' x2='{W - R}' y2='{y:.1f}' stroke='#d9dce1' stroke-dasharray='4 4'/>")
        parts.append(f"<text x='{L - 10}' y='{y + 4:.1f}' font-size='11' fill='#9ca3af' text-anchor='end'>{v}</text>")

    step = max(1, math.ceil(n / 12))
    for i, x in enumerate(xs):
        if i % step == 0:
            parts.append(f"<text x='{x:.1f}' y='{H - 10}' font-size='11' fill='#9ca3af' text-anchor='middle'>{i + 1}</text>")

    peak = max(range(n), key=lambda i: (RISK_SCORE.get(results[i]['risk'], 0), -results[i]['sim']))
    px, py = risk_pts[peak]
    parts.append(f"<rect x='{px - 9:.1f}' y='{py:.1f}' width='18' height='{T + ph - py:.1f}' rx='9' fill='#dbeafe' opacity='0.7'/>")

    parts.append(f"<path d='{smooth_path(sim_pts, T, T + ph)}' fill='none' stroke='#2c6e8f' stroke-width='2.5' stroke-linecap='round'/>")
    parts.append(f"<path d='{smooth_path(risk_pts, T, T + ph)}' fill='none' stroke='#b9605f' stroke-width='2.5' stroke-linecap='round'/>")
    for (x, y) in sim_pts:
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='#2c6e8f'/>")
    for (x, y) in risk_pts:
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='#b9605f'/>")

    label = f"Peak risk: clause {peak + 1}"
    lw = len(label) * 6.4 + 22
    bx = min(max(px - lw / 2, L), W - R - lw)
    by = py - 36 if py - 36 > T else py + 12
    parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5' fill='#fff' stroke='#b9605f' stroke-width='2'/>")
    parts.append(f"<rect x='{bx:.1f}' y='{by:.1f}' width='{lw:.1f}' height='24' rx='7' fill='#fef2f2' stroke='#fca5a5'/>")
    parts.append(f"<text x='{bx + lw / 2:.1f}' y='{by + 16:.1f}' font-size='11' font-weight='500' fill='#b91c1c' text-anchor='middle'>{esc(label)}</text>")
    parts.append("</svg>")
    return ''.join(parts)

def chart_card(results):
    return flat(f"""
    <div class='card'>
        <div class='card-head'>
            <span class='card-title'>Clause Risk Profile</span>
            <span>
                <span class='legend-chip' style='color:#2c6e8f;border-color:#2c6e8f;'>Similarity %</span>
                <span class='legend-chip' style='color:#b9605f;border-color:#b9605f;'>Risk score</span>
            </span>
        </div>
        {trend_chart(results)}
    </div>""")

def table_card(results, limit=6):
    ranked = sorted(results, key=lambda r: (-SEVERITY.get(r['risk'], 0), r['sim']))[:limit]
    rows = ''
    for r in ranked:
        snippet = ' '.join(r['text'].split())
        snippet = snippet[:44] + '…' if len(snippet) > 44 else snippet
        rows += (
            f"<tr><td class='name'>{esc(snippet)}</td>"
            f"<td>{esc(r['type'])}</td>"
            f"<td>{r['idx']}</td>"
            f"<td>{r['sim'] * 100:.0f} %</td>"
            f"<td>{pill(r['risk'], r['risk'])}</td>"
            f"<td>{pill(ACTION_LABELS.get(r['action'], r['action']), ACTION_KIND.get(r['action']))}</td></tr>"
        )
    return flat(f"""
    <div class='card'>
        <div class='card-head'><span class='card-title'>Highest-Risk Clauses</span>{MORE}</div>
        <div class='tbl-wrap'>
        <table class='tbl'>
            <thead><tr><th>Clause</th><th>Type</th><th>No.</th><th>Similarity</th><th>Risk</th><th>Action</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        <div class='foot'>Showing {len(ranked)} of {len(results)} clauses, ranked by risk. See Clause Details for the full list.</div>
    </div>""")

def document_status_card(source_name, source_size, total, accepted):
    ext = os.path.splitext(source_name)[1].lstrip('.').upper()
    ext = ext if ext in FILE_ICON_STYLES else 'TXT'
    bg, fg, border = FILE_ICON_STYLES[ext]
    pct = accepted / total * 100 if total else 0
    return flat(f"""
    <div class='card'>
        <div class='card-head'><span class='card-title'>Document Status</span>{MORE}</div>
        <div class='file-row'>
            <div class='file-ico' style='background:{bg};color:{fg};border-color:{border};'>{ext}</div>
            <div><div class='file-name'>{esc(source_name)}</div><div class='file-meta'>{ext} &middot; {esc(source_size)}</div></div>
        </div>
        <div class='kv'><span>Analyzed</span><b>{datetime.now().strftime('%d %b %Y')}</b></div>
        <div class='kv'><span>Clauses detected</span><b>{total}</b></div>
        <div class='prog-head'>Clauses accepted<span>{pct:.0f}% of total</span></div>
        <div class='prog'><div style='width:{pct:.1f}%;'></div></div>
        <span class='stage'>Stage: rule agent review</span>
    </div>""")

def ai_summary_card(results, overall_risk, flagged_count):
    worst = max(results, key=lambda r: (SEVERITY.get(r['risk'], 0), -r['sim']))
    if worst['action'] == 'ACCEPT':
        reco = "No clauses were flagged by the rule agent for this document."
    else:
        reco = ' '.join(str(w) for w in worst['why'][:2]) or "Review this clause manually."
    return flat(f"""
    <div class='card'>
        <div class='card-head'><span class='card-title'>AI Summary</span>{MORE}</div>
        <div class='kv'><span>Risk zone</span><b>{pill(overall_risk, overall_risk)}</b></div>
        <div class='kv'><span>Top risk clause</span><b>{worst['idx']} &middot; {esc(worst['type'])}</b></div>
        <div class='kv'><span>Flagged</span><b>{flagged_count} of {len(results)} clauses</b></div>
        <div class='reco'>
            <div class='reco-title'>Recommendation</div>
            <div class='reco-body'>{esc(reco)}</div>
        </div>
        <div class='action-bar'><span>Recommended action</span><b>{esc(ACTION_LABELS.get(worst['action'], worst['action']))}</b></div>
    </div>""")

def top_flagged_card(results):
    flagged = sorted([r for r in results if r['action'] != 'ACCEPT'],
                     key=lambda r: (-SEVERITY.get(r['risk'], 0), r['sim']))[:3]
    if flagged:
        rows = ''.join(
            f"<div class='mini-row'><div class='mini-left'><div class='mini-ico'>{r['idx']}</div>"
            f"<div><div class='mini-name'>{esc(r['type'])}</div>"
            f"<div class='mini-meta'>Similarity {r['sim']:.2f}</div></div></div>"
            f"{pill(r['risk'], r['risk'])}</div>"
            for r in flagged
        )
    else:
        rows = "<div class='mini-meta'>No clauses were flagged in this document.</div>"
    return flat(f"""
    <div class='card'>
        <div class='card-head'><span class='card-title'>Top Flagged Clauses</span>{MORE}</div>
        {rows}
    </div>""")

def clause_card(r):
    risk_color = RISK_COLOR_MAP.get(r['risk'], '#6b7280')
    reasons = ''.join(f"<li>{esc(w)}</li>" for w in r['why'])
    sim_pct = max(0.0, min(1.0, r['sim'])) * 100
    return flat(f"""
    <div class='clause' style='--rc:{risk_color};'>
        <div class='clause-head'>
            <span class='clause-title'>Clause {r['idx']} &middot; <span class='type'>{esc(r['type'])}</span></span>
            <span>{pill(r['risk'], r['risk'])} {pill(ACTION_LABELS.get(r['action'], r['action']), ACTION_KIND.get(r['action']))}</span>
        </div>
        <div class='clause-text'>{esc(r['text'])}</div>
        <div class='reasons-title'>Rule agent reasons</div>
        <ul class='reasons'>{reasons}</ul>
        <div class='sim-row'>
            <span>Similarity to market-standard <b>{esc(r['type'])}</b> clause</span>
            <div class='sim-track'><div class='sim-fill' style='width:{sim_pct:.1f}%;'></div></div>
            <b>{r['sim']:.3f}</b>
        </div>
    </div>""")

def nav_button(name, icon, active):
    """Sidebar menu item. The active page is drawn as the primary (white) button."""
    kind = "primary" if active else "secondary"
    try:
        return st.button(name, key=f"nav_{name}", icon=icon, type=kind, use_container_width=True)
    except TypeError:  # older Streamlit without the icon argument
        return st.button(name, key=f"nav_{name}", type=kind, use_container_width=True)

def df_card(title, df, decimals=None, index_header=""):
    """Render a DataFrame as a light table card (same look as the dashboard table)."""
    def fmt(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return esc(v)
        if math.isnan(f):
            return "&ndash;"
        if decimals is None:
            return str(int(f))
        if f.is_integer() and abs(f) > 1:
            return str(int(f))
        return f"{f:.{decimals}f}"

    head = f"<th>{esc(index_header or (df.index.name or ''))}</th>" + ''.join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = ''
    for idx, row in df.iterrows():
        cells = ''.join(f"<td>{fmt(v)}</td>" for v in row.values)
        body += f"<tr><td class='name'>{esc(idx)}</td>{cells}</tr>"
    return flat(f"""
    <div class='card'>
        <div class='card-head'><span class='card-title'>{esc(title)}</span></div>
        <div class='tbl-wrap'>
        <table class='tbl num'>
            <thead><tr>{head}</tr></thead>
            <tbody>{body}</tbody>
        </table>
        </div>
    </div>""")

# =====================================================================
# MAIN APP
# =====================================================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    cuad_file = "CUADv1.json"
    if not os.path.exists(cuad_file):
        st.error(f"Missing dataset file `{cuad_file}` in the workspace. Please download CUADv1.json first.")
        st.stop()

    pipeline = get_trained_pipeline(cuad_file)

    # ---------------- Sidebar ----------------
    source_size = ""
    with st.sidebar:
        st.markdown(f"<div class='brand'>{BRAND_MARK}ClauseInspect</div>", unsafe_allow_html=True)

        st.markdown("<div class='nav-label'>Main</div>", unsafe_allow_html=True)
        if "page" not in st.session_state:
            st.session_state.page = NAV_ITEMS[0]
        for name, icon in zip(NAV_ITEMS, NAV_ICONS):
            if nav_button(name, icon, st.session_state.page == name):
                st.session_state.page = name
                st.rerun()
        page = st.session_state.page

        st.markdown("<div class='nav-label'>Document source</div>", unsafe_allow_html=True)
        doc_source = st.selectbox(
            "Input method",
            ["Upload document", "Built-in demo contract", "Paste text"],
            label_visibility="collapsed"
        )

        raw_contract_text = ""
        source_name = "Built-in Demo"

        if doc_source == "Upload document":
            uploaded_file = st.file_uploader(
                "Upload contract file",
                type=["pdf", "docx", "pptx", "txt", "doc", "ppt"],
                label_visibility="collapsed",
                help="Supports PDF contracts, Word (.docx), PowerPoint (.pptx) and text files."
            )
            if uploaded_file is not None:
                source_name = uploaded_file.name
                source_size = f"{uploaded_file.size / 1024:.1f} KB"
                raw_contract_text = extract_text_from_upload(uploaded_file)
        elif doc_source == "Built-in demo contract":
            raw_contract_text = DEMO_CONTRACT
            source_name = "Built-in 6-Clause Demo Contract"
        else:
            raw_contract_text = st.text_area(
                "Paste contract text",
                height=200,
                label_visibility="collapsed",
                placeholder="Paste contract text with numbered sections or paragraphs..."
            )
            source_name = "Pasted Text"

        st.markdown("<div class='nav-label'>Classifier accuracy</div>", unsafe_allow_html=True)
        rows = ''.join(
            f"<div class='model-row'><span>{esc(m_name)}</span><b>{acc * 100:.2f}%</b></div>"
            for m_name, acc in pipeline['accuracies'].items()
        )
        st.markdown(flat(rows), unsafe_allow_html=True)
        st.markdown(
            f"<div class='side-note' style='margin-top:10px;'>Trained on {pipeline['dataset_size']} CUAD clauses, "
            f"6 categories, 80/20 stratified split.</div>",
            unsafe_allow_html=True
        )

    # ---------------- Page header ----------------
    st.markdown(flat(f"""
    <div class='topbar'>
        <div class='title'>{esc(PAGE_TITLES[page])}</div>
        <div class='doc-chip'>{esc(source_name)}</div>
    </div>"""), unsafe_allow_html=True)

    if not raw_contract_text.strip():
        st.info("Choose a document source in the sidebar (upload a file, use the demo contract, or paste text) to begin the analysis.")
        st.stop()

    if not source_size:
        source_size = f"{len(raw_contract_text):,} characters"

    clauses = split_clauses(raw_contract_text)
    if not clauses:
        st.warning("No clauses longer than 40 characters were identified in the document.")
        st.stop()

    # ---------------- Analysis (unchanged) ----------------
    results = []
    vec = pipeline['vec']
    model = pipeline['model']
    std_vectors = pipeline['std_vectors']

    for idx, clause in enumerate(clauses, 1):
        clean = preprocess(clause)
        c_vec = vec.transform([clean])
        pred_type = model.predict(c_vec)[0]
        sim = float(cosine_similarity(c_vec, std_vectors[pred_type])[0][0])
        facts = extract_facts(clause, pred_type, sim)
        risk, action, why = run_rule_agent(facts)
        results.append({
            'idx': idx,
            'text': clause,
            'type': pred_type,
            'risk': risk,
            'action': action,
            'sim': sim,
            'why': why,
            'facts': facts
        })

    high_count = sum(1 for r in results if r['risk'] == 'High')
    med_count = sum(1 for r in results if r['risk'] == 'Medium')
    low_count = sum(1 for r in results if r['risk'] == 'Low')
    flagged_count = sum(1 for r in results if r['action'] != 'ACCEPT')

    if high_count >= 2:
        overall_risk = "High"
    elif high_count == 1:
        overall_risk = "Medium-High"
    elif med_count > 0:
        overall_risk = "Medium"
    else:
        overall_risk = "Low"

    total = len(results)
    accepted_count = total - flagged_count
    avg_sim = sum(r['sim'] for r in results) / total

    # ---------------- Pages ----------------
    if page == "Dashboard":
        left, right = st.columns([2.15, 1], gap="medium")

        with left:
            kpis = ''.join([
                kpi("Clauses Analyzed", total, "In this document", "flat"),
                kpi("Flagged for Review", flagged_count, f"{flagged_count / total * 100:.0f}% of clauses",
                    "down" if flagged_count else "up"),
                kpi("High Risk Terms", high_count, "Critical clauses" if high_count else "None found",
                    "down" if high_count else "up"),
                kpi("Avg. Similarity", f"{avg_sim * 100:.0f}%", "To market standard", "up"),
            ])
            st.markdown(f"<div class='kpi-grid'>{kpis}</div>", unsafe_allow_html=True)
            st.markdown(chart_card(results) + table_card(results), unsafe_allow_html=True)

        with right:
            st.markdown(
                document_status_card(source_name, source_size, total, accepted_count)
                + ai_summary_card(results, overall_risk, flagged_count)
                + top_flagged_card(results),
                unsafe_allow_html=True
            )

    elif page == "Clause Details":
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            risk_options = ["High", "Medium-High", "Medium", "Low"]
            risk_filter = st.multiselect("Filter by risk level", options=risk_options, default=risk_options)
        with filter_col2:
            action_filter = st.multiselect(
                "Filter by action",
                options=list(ACTION_LABELS.keys()),
                default=list(ACTION_LABELS.keys()),
                format_func=lambda k: ACTION_LABELS[k]
            )

        filtered_results = [r for r in results if r['risk'] in risk_filter and r['action'] in action_filter]
        st.caption(f"Showing {len(filtered_results)} of {len(results)} clauses.")

        for r in filtered_results:
            st.markdown(clause_card(r), unsafe_allow_html=True)

    elif page == "Classifier Metrics":
        rep_df = pd.DataFrame(pipeline['report']).transpose()
        st.markdown(df_card("Logistic Regression (primary classifier)", rep_df, decimals=3), unsafe_allow_html=True)

        classes = pipeline.get('class_names') or sorted(list(TARGET_CATEGORIES))
        cm_df = pd.DataFrame(
            pipeline['confusion_matrix'],
            index=[f"True: {c}" for c in classes],
            columns=[f"Pred: {c}" for c in classes]
        )
        st.markdown(df_card("Confusion matrix", cm_df), unsafe_allow_html=True)

    else:
        crosstab = pipeline['crosstab'].copy()
        crosstab.columns = [f"Cluster {c}" for c in crosstab.columns]
        st.markdown(
            df_card("K-Means clustering (k = 3) on the TF-IDF matrix", crosstab, index_header="True clause type"),
            unsafe_allow_html=True
        )
        st.caption("How clauses naturally group in vector space compared with their true legal category.")

if __name__ == '__main__':
    main()
"""TalentVector — semantic resume ranking."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from matching import rank
from parsing import parse_uploaded_file, parse_text
from summary import fit_summary

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentVector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Design system ─────────────────────────────────────────────────────────────
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background:
        radial-gradient(ellipse at 15% 25%, rgba(99,102,241,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 75%, rgba(139,92,246,0.13) 0%, transparent 55%),
        linear-gradient(160deg, #08081a 0%, #0e0e28 55%, #080818 100%) !important;
    color: #e2e8f0 !important;
    min-height: 100vh;
}

[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Main content container ── */
[data-testid="stAppViewBlockContainer"] {
    max-width: 1120px !important;
    padding: 2rem 2rem 4rem !important;
}

/* ── Typography scale ── */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.02em;
}

/* ── Glass card utility ── */
.glass {
    background: rgba(255,255,255,0.055);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    box-shadow:
        0 8px 40px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ── Header ── */
.tv-header {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 2.5rem;
}
.tv-wordmark {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.25rem;
    line-height: 1;
}
.tv-tagline {
    font-size: 0.95rem;
    font-weight: 400;
    color: rgba(148,163,184,0.9);
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.65);
    margin: 0 0 0.75rem;
}

/* ── Input panels ── */
.input-panel {
    padding: 1.5rem;
    height: 100%;
}

.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99,102,241,0.55) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
.stTextArea > label { display: none !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.45) !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(148,163,184,0.8) !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stFileUploader"] small {
    color: rgba(100,116,139,0.7) !important;
    font-size: 0.75rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    margin-bottom: 1rem !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(148,163,184,0.6) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 5px 14px !important;
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.09) !important;
    color: #e2e8f0 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}
.stTextInput > label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    border-radius: 10px !important;
    border: none !important;
    transition: all 0.2s !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
    color: white !important;
    padding: 0.65rem 2rem !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 32px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(203,213,225,0.9) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    padding: 0.6rem 1.25rem !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.18) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 2rem 0 !important;
}

/* ── Caption / small text ── */
.stCaption, .stMarkdown p small {
    color: rgba(100,116,139,0.8) !important;
    font-size: 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stMetricLabel"] {
    color: rgba(148,163,184,0.7) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    overflow: hidden !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
</style>
"""

CARD_CSS = """
<style>
/* ── Candidate result cards ── */
.candidate-card {
    background: rgba(255,255,255,0.048);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.07);
    transition: border-color 0.2s;
}
.candidate-card:hover {
    border-color: rgba(99,102,241,0.3);
}

/* Hero row */
.card-hero {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem 1.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.card-rank {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(99,102,241,0.8);
    min-width: 28px;
}
.card-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
    flex: 1;
}
.card-score {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f1f5f9;
    min-width: 72px;
    text-align: right;
}
.card-score-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: rgba(148,163,184,0.6);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: right;
    margin-top: 2px;
}
.badge {
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
}
.badge-strong {
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.25);
}
.badge-partial {
    background: rgba(245,158,11,0.12);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.22);
}
.badge-weak {
    background: rgba(239,68,68,0.10);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.20);
}

/* Score breakdown */
.card-breakdown {
    padding: 1.1rem 1.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.breakdown-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(100,116,139,0.7);
    margin-bottom: 0.85rem;
}
.breakdown-row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0.6rem;
}
.breakdown-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: rgba(148,163,184,0.85);
    min-width: 130px;
}
.breakdown-track {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
}
.breakdown-fill-doc {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #6366f1, #818cf8);
}
.breakdown-fill-skill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #10b981, #34d399);
}
.breakdown-right {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    min-width: 130px;
    justify-content: flex-end;
}
.breakdown-pct {
    font-size: 0.78rem;
    font-weight: 600;
    color: #cbd5e1;
    min-width: 36px;
    text-align: right;
}
.breakdown-contrib {
    font-size: 0.72rem;
    font-weight: 500;
    color: rgba(100,116,139,0.7);
    min-width: 50px;
    text-align: right;
}
.score-driver {
    margin-top: 0.75rem;
    font-size: 0.78rem;
    color: rgba(148,163,184,0.65);
    line-height: 1.5;
    font-style: italic;
}

/* Skills section */
.card-skills {
    padding: 1.1rem 1.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.skills-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
}
.skills-col-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.skills-col-label.matched { color: rgba(52,211,153,0.7); }
.skills-col-label.missing { color: rgba(248,113,113,0.7); }
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.chip {
    display: inline-flex;
    align-items: center;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    line-height: 1.4;
}
.chip-matched {
    background: rgba(16,185,129,0.10);
    color: #6ee7b7;
    border: 1px solid rgba(16,185,129,0.18);
}
.chip-missing {
    background: rgba(239,68,68,0.08);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.15);
}
.chip-extra {
    background: rgba(99,102,241,0.08);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.15);
}

/* Fit summary */
.card-summary {
    padding: 1.1rem 1.75rem;
}
.summary-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(100,116,139,0.7);
    margin-bottom: 0.5rem;
}
.summary-text {
    font-size: 0.84rem;
    color: rgba(203,213,225,0.85);
    line-height: 1.65;
}

/* Results header */
.results-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.results-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0;
}
.results-count {
    font-size: 0.78rem;
    color: rgba(100,116,139,0.7);
    font-weight: 500;
}
.disclaimer {
    font-size: 0.72rem;
    color: rgba(100,116,139,0.6);
    margin-top: 0.3rem;
    margin-bottom: 1.5rem;
    font-style: italic;
    line-height: 1.5;
}

/* Input section headers */
.panel-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.7);
    margin-bottom: 1rem;
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ─── Helper functions ──────────────────────────────────────────────────────────
EXAMPLES = Path(__file__).parent / "examples"


def _load_examples() -> tuple[str, dict[str, str]]:
    jd = (EXAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
    resumes = {}
    for f in sorted(EXAMPLES.glob("candidate_*.txt")):
        label = f.stem.replace("candidate_", "").replace("_", " ").title()
        resumes[label] = f.read_text(encoding="utf-8")
    return jd, resumes


def _badge_html(action: str) -> str:
    cls = (
        "badge-strong" if "Strong" in action
        else "badge-partial" if "Partial" in action
        else "badge-weak"
    )
    return f'<span class="badge {cls}">{action}</span>'


def _chips_html(skills: list[str], kind: str, limit: int = 20) -> str:
    if not skills:
        return '<span style="font-size:0.75rem;color:rgba(100,116,139,0.5)">None detected</span>'
    return "".join(
        f'<span class="chip chip-{kind}">{s}</span>'
        for s in skills[:limit]
    )


def _bar(pct: float, kind: str) -> str:
    w = min(max(pct, 0), 100)
    return (
        f'<div class="breakdown-track">'
        f'<div class="breakdown-fill-{kind}" style="width:{w}%"></div>'
        f'</div>'
    )


def _render_card(rank_num: int, r: dict) -> str:
    pct = round(r["score"] * 100, 1)
    bd = r["score_breakdown"]
    n_matched = len(r["matched_skills"])
    n_jd = r["jd_skill_count"]

    matched_chips = _chips_html(r["matched_skills"], "matched", 18)
    missing_chips = _chips_html(r["missing_skills"][:18], "missing", 18)

    summary_html = (
        f'<div class="card-summary">'
        f'<div class="summary-label">Fit assessment</div>'
        f'<div class="summary-text">{r.get("summary", "")}</div>'
        f'</div>'
        if r.get("summary") else ""
    )

    return f"""
<div class="candidate-card">
  <div class="card-hero">
    <span class="card-rank">#{rank_num}</span>
    <span class="card-name">{r['name']}</span>
    {_badge_html(r['next_action'])}
    <div>
      <div class="card-score">{pct}%</div>
      <div class="card-score-label">Match score</div>
    </div>
  </div>

  <div class="card-breakdown">
    <div class="breakdown-title">Score breakdown</div>

    <div class="breakdown-row">
      <span class="breakdown-label">Content alignment</span>
      {_bar(bd['doc_pct'], 'doc')}
      <div class="breakdown-right">
        <span class="breakdown-pct">{bd['doc_pct']}%</span>
        <span class="breakdown-contrib">+{bd['doc_pts']} pts</span>
      </div>
    </div>

    <div class="breakdown-row">
      <span class="breakdown-label">Skill coverage &nbsp;<span style="color:rgba(100,116,139,0.55);font-weight:400">{n_matched}/{n_jd}</span></span>
      {_bar(bd['skill_pct'], 'skill')}
      <div class="breakdown-right">
        <span class="breakdown-pct">{bd['skill_pct']}%</span>
        <span class="breakdown-contrib">+{bd['skill_pts']} pts</span>
      </div>
    </div>

    <div class="score-driver">{r.get('score_driver', '')}</div>
  </div>

  <div class="card-skills">
    <div class="skills-grid">
      <div>
        <div class="skills-col-label matched">Matched skills</div>
        <div class="chips">{matched_chips}</div>
      </div>
      <div>
        <div class="skills-col-label missing">Missing skills</div>
        <div class="chips">{missing_chips}</div>
      </div>
    </div>
  </div>

  {summary_html}
</div>
"""


# ─── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "resume_map" not in st.session_state:
    st.session_state["resume_map"] = {}

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tv-header">
  <div class="tv-wordmark">TalentVector</div>
  <div class="tv-tagline">
    Semantic resume ranking — contextual skill matching, not keyword counting
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Demo button ───────────────────────────────────────────────────────────────
if st.button("Load example", type="secondary"):
    jd_ex, res_ex = _load_examples()
    st.session_state["jd_text"] = jd_ex
    st.session_state["resume_map"] = res_ex
    st.session_state["results"] = None

st.divider()

# ─── Input columns ─────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="panel-title">Job Description</div>', unsafe_allow_html=True)
    jd_tab_paste, jd_tab_upload = st.tabs(["Paste text", "Upload file"])
    with jd_tab_paste:
        jd_paste = st.text_area(
            "_jd_paste",
            value=st.session_state["jd_text"],
            height=300,
            placeholder="Paste the full job description here — responsibilities, requirements, nice-to-haves.",
            label_visibility="collapsed",
        )
    with jd_tab_upload:
        jd_file = st.file_uploader(
            "Upload JD — PDF or DOCX",
            type=["pdf", "docx"],
            key="jd_file_upload",
        )

with right:
    st.markdown('<div class="panel-title">Candidate Resumes</div>', unsafe_allow_html=True)
    res_tab_upload, res_tab_paste = st.tabs(["Upload files", "Paste text"])
    with res_tab_upload:
        res_files = st.file_uploader(
            "Upload resumes — PDF or DOCX, up to 10 files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="res_file_upload",
        )
    with res_tab_paste:
        res_paste_name = st.text_input(
            "_res_name",
            value="Candidate",
            placeholder="Candidate name",
            label_visibility="collapsed",
        )
        res_paste_text = st.text_area(
            "_res_paste",
            height=245,
            placeholder="Paste resume text here.",
            label_visibility="collapsed",
        )


# ─── Resolve inputs ────────────────────────────────────────────────────────────
def _resolve_jd() -> str:
    if jd_file is not None:
        return parse_uploaded_file(jd_file.name, jd_file.read())
    return parse_text(jd_paste or st.session_state.get("jd_text", ""))


def _resolve_resumes() -> dict[str, str]:
    result = dict(st.session_state.get("resume_map", {}))

    if res_files:
        result = {}
        for f in res_files[:10]:
            name = f.name.rsplit(".", 1)[0]
            result[name] = parse_uploaded_file(f.name, f.read())

    if res_paste_text and res_paste_text.strip():
        result[res_paste_name or "Candidate"] = parse_text(res_paste_text)

    return result


# ─── CTA ────────────────────────────────────────────────────────────────────────
st.divider()

gemini_active = bool(os.environ.get("GEMINI_API_KEY"))
if not gemini_active:
    try:
        gemini_active = bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        pass

if gemini_active:
    st.caption(
        "AI summaries active — resume text excerpts are sent to Google Generative AI for fit assessment only. "
        "No data is stored."
    )
else:
    st.caption(
        "Running without an API key — fit assessments are template-generated from matched/missing skills. "
        "Set GEMINI_API_KEY in Streamlit secrets to enable AI summaries."
    )

rank_btn = st.button("Rank candidates", type="primary")

if rank_btn:
    jd = _resolve_jd()
    resumes = _resolve_resumes()

    if not jd.strip():
        st.error("Provide a job description before ranking.")
    elif not resumes:
        st.error("Upload or paste at least one resume.")
    else:
        with st.spinner("Embedding documents and ranking candidates…"):
            ranked = rank(jd, resumes)
            for r in ranked:
                r["summary"] = fit_summary(
                    jd, resumes[r["name"]], r["matched_skills"], r["missing_skills"]
                )

        st.session_state["results"] = ranked
        st.session_state["jd_text"] = jd

# ─── Results ──────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    st.divider()

    n = len(results)
    st.markdown(f"""
    <div class="results-header">
      <span class="results-title">Ranked Shortlist</span>
      <span class="results-count">{n} candidate{"s" if n != 1 else ""}</span>
    </div>
    <div class="disclaimer">
      Scores reflect semantic similarity and skill coverage — heuristic estimates, not a validated hiring metric.
      Use as a triage signal; review the score breakdown for each candidate before making advancement decisions.
    </div>
    """, unsafe_allow_html=True)

    for i, r in enumerate(results):
        st.markdown(_render_card(i + 1, r), unsafe_allow_html=True)

    # Summary table
    st.divider()
    st.markdown('<div class="panel-title">Comparison table</div>', unsafe_allow_html=True)
    df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Candidate": r["name"],
            "Score": f"{round(r['score']*100,1)}%",
            "Content align": f"{r['score_breakdown']['doc_pct']}%",
            "Skill coverage": f"{len(r['matched_skills'])}/{r['jd_skill_count']}",
            "Top missing": ", ".join(r["missing_skills"][:3]) or "—",
            "Recommendation": r["next_action"],
        }
        for i, r in enumerate(results)
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

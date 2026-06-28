"""TalentVector — semantic resume ranking."""

import os
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from matching import rank
from parsing import parse_uploaded_file, parse_text
from summary import fit_summary

st.set_page_config(page_title="TalentVector", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..800;1,14..32,300..800&display=swap');

/* ── tokens ── */
:root {
  --bg:          #070712;
  --bg-2:        rgba(255,255,255,0.032);
  --bg-3:        rgba(255,255,255,0.058);
  --border:      rgba(255,255,255,0.078);
  --border-hi:   rgba(255,255,255,0.16);
  --accent:      #6366f1;
  --accent-dim:  rgba(99,102,241,0.18);
  --accent-glow: rgba(99,102,241,0.38);
  --t1:          #f1f5f9;
  --t2:          #94a3b8;
  --t3:          #475569;
  --success:     #10b981;
  --warn:        #f59e0b;
  --danger:      #ef4444;
  --radius-sm:   8px;
  --radius-md:   12px;
  --radius-lg:   18px;
}

/* ── base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  font-family: 'Inter', system-ui, sans-serif !important;
  background:
    radial-gradient(ellipse 55% 45% at 8% 15%,  rgba(99,102,241,0.22) 0%, transparent 55%),
    radial-gradient(ellipse 45% 55% at 92% 85%, rgba(139,92,246,0.16) 0%, transparent 55%),
    var(--bg) !important;
  color: var(--t1) !important;
}

[data-testid="stHeader"] {
  background: rgba(7,7,18,0.8) !important;
  backdrop-filter: blur(16px) !important;
  border-bottom: 1px solid var(--border) !important;
}

#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

[data-testid="stAppViewBlockContainer"] {
  max-width: 1080px !important;
  padding: 0 2rem 5rem !important;
}

/* ── force Inter everywhere ── */
html, body, input, textarea, button, select, p, span, div, label, td, th, a {
  font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }

/* ───────────────────────────────────────────
   STREAMLIT COMPONENT OVERRIDES
   (only the ones that survive base=dark)
─────────────────────────────────────────── */

/* Textarea */
.stTextArea textarea {
  background: rgba(5,5,20,0.75) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: #c8d4e8 !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  caret-color: var(--accent) !important;
  transition: border-color 140ms, box-shadow 140ms !important;
  resize: vertical !important;
}
.stTextArea textarea:focus {
  border-color: rgba(99,102,241,0.6) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 1px 8px rgba(0,0,0,0.4) !important;
  outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--t3) !important; }
.stTextArea label { display: none !important; }

/* Text input */
.stTextInput input {
  background: rgba(5,5,20,0.75) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: #c8d4e8 !important;
  font-size: 13.5px !important;
  transition: border-color 140ms, box-shadow 140ms !important;
}
.stTextInput input:focus {
  border-color: rgba(99,102,241,0.6) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
  outline: none !important;
}
.stTextInput label { display: none !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(5,5,20,0.60) !important;
  border: 1.5px dashed rgba(99,102,241,0.28) !important;
  border-radius: var(--radius-md) !important;
  transition: border-color 140ms, background 140ms !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(99,102,241,0.55) !important;
  background: rgba(10,10,40,0.65) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--t3) !important;
  font-size: 12.5px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
  color: rgba(71,85,105,0.6) !important;
  font-size: 11.5px !important;
}
.stFileUploader label {
  color: var(--t2) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 3px !important;
  gap: 2px !important;
  margin-bottom: 12px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--t3) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  border-radius: 5px !important;
  padding: 5px 14px !important;
  border: none !important;
  transition: color 120ms, background 120ms !important;
  letter-spacing: 0.01em !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent-dim) !important;
  color: #a5b4fc !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [role="tabpanel"] { padding-top: 0 !important; }

/* Buttons */
.stButton > button {
  font-size: 13.5px !important;
  font-weight: 600 !important;
  letter-spacing: 0.015em !important;
  border-radius: var(--radius-sm) !important;
  border: none !important;
  transition: transform 140ms, box-shadow 140ms, background 140ms !important;
  cursor: pointer !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
  color: #fff !important;
  padding: 10px 28px !important;
  box-shadow: 0 4px 16px rgba(99,102,241,0.40), 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 28px rgba(99,102,241,0.55), 0 2px 6px rgba(0,0,0,0.3) !important;
}
.stButton > button[kind="primary"]:active {
  transform: translateY(0) scale(0.98) !important;
  box-shadow: 0 2px 8px rgba(99,102,241,0.30) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--bg-2) !important;
  color: var(--t2) !important;
  border: 1px solid var(--border) !important;
  padding: 8px 16px !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--bg-3) !important;
  border-color: var(--border-hi) !important;
  color: var(--t1) !important;
  transform: translateY(-1px) !important;
}

/* Divider */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 24px 0 !important;
}

/* Caption */
.stCaption {
  color: var(--t3) !important;
  font-size: 12px !important;
  line-height: 1.5 !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Error / success */
.stAlert { border-radius: var(--radius-sm) !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE-SPECIFIC STYLES (header, cards, layout chrome)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── HEADER ── */
.tv-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2.5rem 0 1.75rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.tv-left {}
.tv-wordmark {
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: -0.045em;
  line-height: 1;
  background: linear-gradient(125deg, #a5b4fc 0%, #818cf8 35%, #c084fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: block;
  margin-bottom: 6px;
}
.tv-tagline {
  font-size: 13px;
  color: var(--t3);
  font-weight: 400;
  letter-spacing: 0.01em;
}
.tv-badge {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #a5b4fc;
  background: var(--accent-dim);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 20px;
  padding: 5px 12px;
}

/* ── SECTION LABEL ── */
.sec-label {
  display: block;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 12px;
}

/* ── RESULTS HEADER ── */
.results-bar {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 6px;
}
.results-title { font-size: 15px; font-weight: 700; color: var(--t1); letter-spacing: -0.02em; }
.results-count { font-size: 12px; color: var(--t3); font-weight: 500; }
.disclaimer {
  font-size: 11.5px;
  color: var(--t3);
  font-style: italic;
  line-height: 1.55;
  margin-bottom: 20px;
  opacity: 0.75;
}

/* ═══════════════════════════════
   RESULT CARD
═══════════════════════════════ */
.rcard {
  background: rgba(255,255,255,0.040);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  margin-bottom: 14px;
  box-shadow:
    0 2px 1px rgba(255,255,255,0.04) inset,
    0 -1px 0 rgba(0,0,0,0.2) inset,
    0 8px 48px rgba(0,0,0,0.55);
  transition: border-color 180ms, box-shadow 180ms;
}
.rcard:hover {
  border-color: rgba(99,102,241,0.32);
  box-shadow: 0 2px 1px rgba(255,255,255,0.05) inset, 0 12px 56px rgba(0,0,0,0.6);
}

/* ── hero row ── */
.rcard-hero {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
}

/* score ring */
.ring-wrap {
  position: relative;
  width: 68px;
  height: 68px;
  flex-shrink: 0;
}
.ring-wrap svg { position: absolute; top: 0; left: 0; }
.ring-center {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0;
}
.ring-pct {
  font-size: 15px;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
}
.ring-lbl {
  font-size: 8.5px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--t3);
  margin-top: 2px;
}

/* name/meta */
.hero-meta { flex: 1; min-width: 0; }
.hero-rank {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(99,102,241,0.7);
  margin-bottom: 5px;
}
.hero-name {
  font-size: 16px;
  font-weight: 700;
  color: var(--t1);
  letter-spacing: -0.015em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* badge */
.badge {
  flex-shrink: 0;
  padding: 5px 13px;
  border-radius: 20px;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  white-space: nowrap;
}
.b-strong  { background: rgba(16,185,129,0.13); color: #34d399; border: 1px solid rgba(16,185,129,0.22); }
.b-partial { background: rgba(245,158,11,0.11); color: #fbbf24; border: 1px solid rgba(245,158,11,0.20); }
.b-weak    { background: rgba(239,68,68,0.09);  color: #f87171; border: 1px solid rgba(239,68,68,0.17); }

/* ── breakdown ── */
.rcard-bd {
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
}
.bd-hd {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 12px;
  opacity: 0.7;
}
.bd-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.bd-label {
  font-size: 12.5px;
  font-weight: 500;
  color: var(--t2);
  min-width: 130px;
  flex-shrink: 0;
}
.bd-track {
  flex: 1;
  height: 3.5px;
  background: rgba(255,255,255,0.055);
  border-radius: 2px;
  overflow: hidden;
}
.fill-d { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #6366f1, #818cf8); }
.fill-s { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #10b981, #34d399); }
.bd-stats {
  display: flex;
  gap: 10px;
  min-width: 108px;
  justify-content: flex-end;
  flex-shrink: 0;
}
.bd-pct { font-size: 12px; font-weight: 600; color: #cbd5e1; min-width: 34px; text-align: right; }
.bd-pts { font-size: 11px; color: var(--t3); min-width: 48px; text-align: right; }
.bd-driver {
  margin-top: 10px;
  font-size: 12px;
  color: var(--t3);
  line-height: 1.55;
  font-style: italic;
  padding-left: 2px;
  border-left: 2px solid rgba(255,255,255,0.06);
  padding-left: 10px;
}

/* ── skills ── */
.rcard-sk {
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
}
.sk-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.sk-col-hd {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 8px;
  display: block;
}
.skh-m { color: rgba(52,211,153,0.65); }
.skh-x { color: rgba(248,113,113,0.65); }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip {
  display: inline-flex;
  padding: 3px 8px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 500;
  line-height: 1.4;
  letter-spacing: 0.005em;
}
.chm { background: rgba(16,185,129,0.08); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.13); }
.chx { background: rgba(239,68,68,0.07);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.12); }
.no-sk { font-size: 11.5px; color: var(--t3); font-style: italic; opacity: 0.6; }

/* ── summary ── */
.rcard-sum {
  padding: 14px 24px 20px;
}
.sum-hd {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 7px;
  opacity: 0.7;
}
.sum-body {
  font-size: 13px;
  color: rgba(203,213,225,0.78);
  line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = Path(__file__).parent / "examples"


def _load_examples():
    jd = (EXAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
    return jd, {
        f.stem.replace("candidate_", "").replace("_", " ").title(): f.read_text(encoding="utf-8")
        for f in sorted(EXAMPLES.glob("candidate_*.txt"))
    }


def _score_ring(pct: float) -> str:
    r, cx = 28, 34
    circ = round(2 * math.pi * r, 2)
    offset = round(circ * (1 - min(max(pct, 0), 100) / 100), 2)
    if pct >= 61:
        color, glow = "#10b981", "rgba(16,185,129,0.50)"
    elif pct >= 42:
        color, glow = "#f59e0b", "rgba(245,158,11,0.50)"
    else:
        color, glow = "#ef4444", "rgba(239,68,68,0.45)"
    return f"""
<div class="ring-wrap">
  <svg width="68" height="68" viewBox="0 0 68 68">
    <circle cx="{cx}" cy="{cx}" r="{r}" fill="none"
      stroke="rgba(255,255,255,0.055)" stroke-width="4"/>
    <circle cx="{cx}" cy="{cx}" r="{r}" fill="none"
      stroke="{color}" stroke-width="4"
      stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
      stroke-linecap="round" transform="rotate(-90 {cx} {cx})"
      style="filter:drop-shadow(0 0 5px {glow});transition:stroke-dashoffset 0.6s ease"/>
  </svg>
  <div class="ring-center">
    <span class="ring-pct" style="color:{color}">{pct}%</span>
    <span class="ring-lbl">score</span>
  </div>
</div>"""


def _badge(action: str) -> str:
    cls = "b-strong" if "Strong" in action else "b-partial" if "Partial" in action else "b-weak"
    return f'<span class="badge {cls}">{action}</span>'


def _chips(skills: list, kind: str, limit: int = 20) -> str:
    if not skills:
        return '<span class="no-sk">None detected</span>'
    return "".join(f'<span class="chip ch{kind}">{s}</span>' for s in skills[:limit])


def _bar(pct: float, fill: str) -> str:
    w = min(max(float(pct), 0), 100)
    return f'<div class="bd-track"><div class="fill-{fill}" style="width:{w}%"></div></div>'


def _card(i: int, r: dict) -> str:
    pct = round(r["score"] * 100, 1)
    bd  = r["score_breakdown"]
    nm, nj = len(r["matched_skills"]), r["jd_skill_count"]
    summary_html = (
        f'<div class="rcard-sum"><div class="sum-hd">Fit assessment</div>'
        f'<div class="sum-body">{r["summary"]}</div></div>'
    ) if r.get("summary") else ""

    return f"""
<div class="rcard">
  <div class="rcard-hero">
    {_score_ring(pct)}
    <div class="hero-meta">
      <div class="hero-rank">Rank #{i}</div>
      <div class="hero-name">{r['name']}</div>
    </div>
    {_badge(r['next_action'])}
  </div>

  <div class="rcard-bd">
    <div class="bd-hd">Score breakdown</div>
    <div class="bd-row">
      <span class="bd-label">Content alignment</span>
      {_bar(bd['doc_pct'], 'd')}
      <div class="bd-stats">
        <span class="bd-pct">{bd['doc_pct']}%</span>
        <span class="bd-pts">+{bd['doc_pts']} pts</span>
      </div>
    </div>
    <div class="bd-row">
      <span class="bd-label">Skill coverage&ensp;<span style="color:var(--t3);font-weight:400;font-size:11px">{nm}/{nj}</span></span>
      {_bar(bd['skill_pct'], 's')}
      <div class="bd-stats">
        <span class="bd-pct">{bd['skill_pct']}%</span>
        <span class="bd-pts">+{bd['skill_pts']} pts</span>
      </div>
    </div>
    <div class="bd-driver">{r.get('score_driver', '')}</div>
  </div>

  <div class="rcard-sk">
    <div class="sk-grid">
      <div>
        <span class="sk-col-hd skh-m">Matched skills</span>
        <div class="chips">{_chips(r['matched_skills'], 'm')}</div>
      </div>
      <div>
        <span class="sk-col-hd skh-x">Missing skills</span>
        <div class="chips">{_chips(r['missing_skills'][:20], 'x')}</div>
      </div>
    </div>
  </div>

  {summary_html}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("results", None), ("jd_text", ""), ("resume_map", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tv-header">
  <div class="tv-left">
    <span class="tv-wordmark">TalentVector</span>
    <span class="tv-tagline">Semantic resume ranking — contextual skill matching, not keyword counting</span>
  </div>
  <span class="tv-badge">Embedding-based</span>
</div>
""", unsafe_allow_html=True)

col_btn, _ = st.columns([1, 5])
with col_btn:
    if st.button("Load example", type="secondary", use_container_width=True):
        jd_ex, res_ex = _load_examples()
        st.session_state["jd_text"] = jd_ex
        st.session_state["resume_map"] = res_ex
        st.session_state["results"] = None

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<span class="sec-label">Job Description</span>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Paste text", "Upload file"])
    with t1:
        jd_paste = st.text_area(
            "_jd",
            value=st.session_state["jd_text"],
            height=300,
            placeholder="Paste the full job description — responsibilities, requirements, qualifications.",
            label_visibility="collapsed",
        )
    with t2:
        jd_file = st.file_uploader(
            "Upload JD",
            type=["pdf", "docx"],
            key="jd_up",
            label_visibility="collapsed",
        )

with right:
    st.markdown('<span class="sec-label">Candidate Resumes</span>', unsafe_allow_html=True)
    t3, t4 = st.tabs(["Upload files", "Paste text"])
    with t3:
        res_files = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="res_up",
            label_visibility="collapsed",
        )
    with t4:
        res_name = st.text_input(
            "_name",
            value="Candidate",
            placeholder="Candidate name",
            label_visibility="collapsed",
        )
        res_paste = st.text_area(
            "_resume",
            height=248,
            placeholder="Paste resume text here.",
            label_visibility="collapsed",
        )


def _resolve_jd() -> str:
    if jd_file:
        return parse_uploaded_file(jd_file.name, jd_file.read())
    return parse_text(jd_paste or st.session_state.get("jd_text", ""))


def _resolve_resumes() -> dict:
    base = dict(st.session_state.get("resume_map", {}))
    if res_files:
        base = {f.name.rsplit(".", 1)[0]: parse_uploaded_file(f.name, f.read()) for f in res_files[:10]}
    if res_paste and res_paste.strip():
        base[res_name or "Candidate"] = parse_text(res_paste)
    return base


# ─────────────────────────────────────────────────────────────────────────────
#  CTA
# ─────────────────────────────────────────────────────────────────────────────
st.divider()

gemini_on = bool(os.environ.get("GEMINI_API_KEY"))
if not gemini_on:
    try:
        gemini_on = bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        pass

st.caption(
    "AI fit summaries active — resume excerpts sent to Google Generative AI for assessment only." if gemini_on
    else "Template fit summaries active. Set GEMINI_API_KEY in Streamlit secrets to enable AI-generated assessments."
)

cta_col, _ = st.columns([2, 5])
with cta_col:
    run = st.button("Rank candidates", type="primary", use_container_width=True)

if run:
    jd      = _resolve_jd()
    resumes = _resolve_resumes()
    if not jd.strip():
        st.error("Provide a job description before ranking.")
    elif not resumes:
        st.error("Upload or paste at least one resume.")
    else:
        with st.spinner("Embedding documents and ranking…"):
            ranked = rank(jd, resumes)
            for r in ranked:
                r["summary"] = fit_summary(
                    jd, resumes[r["name"]], r["matched_skills"], r["missing_skills"]
                )
        st.session_state["results"] = ranked
        st.session_state["jd_text"] = jd

# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    st.divider()
    n = len(results)
    st.markdown(f"""
<div class="results-bar">
  <span class="results-title">Ranked Shortlist</span>
  <span class="results-count">{n} candidate{"s" if n != 1 else ""}</span>
</div>
<div class="disclaimer">
  Scores are semantic similarity estimates — not a validated hiring metric.
  Review the score breakdown per candidate before making advancement decisions.
</div>
""", unsafe_allow_html=True)

    for i, r in enumerate(results):
        st.markdown(_card(i + 1, r), unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="sec-label">Comparison table</span>', unsafe_allow_html=True)
    df = pd.DataFrame([{
        "Rank": i + 1,
        "Candidate": r["name"],
        "Score": f"{round(r['score']*100, 1)}%",
        "Content align": f"{r['score_breakdown']['doc_pct']}%",
        "Skill coverage": f"{len(r['matched_skills'])}/{r['jd_skill_count']}",
        "Top gaps": ", ".join(r["missing_skills"][:3]) or "—",
        "Recommendation": r["next_action"],
    } for i, r in enumerate(results)])
    st.dataframe(df, use_container_width=True, hide_index=True)

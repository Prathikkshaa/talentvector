"""TalentVector — semantic resume↔JD matching UI."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from matching import rank
from parsing import parse_uploaded_file, parse_text
from summary import fit_summary

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentVector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .tv-header { font-size: 2.4rem; font-weight: 800; margin-bottom: 0; }
    .tv-sub    { color: #6b7280; margin-top: 0; margin-bottom: 1.5rem; }
    .badge-strong { background:#dcfce7; color:#166534; padding:3px 10px;
                    border-radius:12px; font-size:0.82rem; font-weight:600; }
    .badge-partial { background:#fef9c3; color:#854d0e; padding:3px 10px;
                     border-radius:12px; font-size:0.82rem; font-weight:600; }
    .badge-weak { background:#fee2e2; color:#991b1b; padding:3px 10px;
                  border-radius:12px; font-size:0.82rem; font-weight:600; }
    .chip-match  { background:#dcfce7; color:#166534; display:inline-block;
                   padding:2px 8px; border-radius:10px; margin:2px;
                   font-size:0.78rem; }
    .chip-miss   { background:#fee2e2; color:#991b1b; display:inline-block;
                   padding:2px 8px; border-radius:10px; margin:2px;
                   font-size:0.78rem; }
    .chip-extra  { background:#e0f2fe; color:#0369a1; display:inline-block;
                   padding:2px 8px; border-radius:10px; margin:2px;
                   font-size:0.78rem; }
    .rank-num    { font-size:1.8rem; font-weight:800; color:#6366f1; }
    .score-big   { font-size:1.4rem; font-weight:700; }
    .disclaimer  { font-size:0.78rem; color:#9ca3af; margin-top:0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="tv-header">🎯 TalentVector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="tv-sub">Semantic resume ranking — paste or upload one job description '
    "and up to 10 resumes to get a ranked, explained shortlist in seconds.</p>",
    unsafe_allow_html=True,
)

# ── Helper: load example files ────────────────────────────────────────────────
EXAMPLES = Path(__file__).parent / "examples"


def _load_examples() -> tuple[str, dict[str, str]]:
    jd = (EXAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
    resumes = {}
    for f in sorted(EXAMPLES.glob("candidate_*.txt")):
        label = f.stem.replace("candidate_", "").replace("_", " ").title()
        resumes[label] = f.read_text(encoding="utf-8")
    return jd, resumes


# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "resume_map" not in st.session_state:
    st.session_state["resume_map"] = {}

# ── Demo button ───────────────────────────────────────────────────────────────
demo_col, _ = st.columns([1, 4])
with demo_col:
    if st.button("⚡ Load example", use_container_width=True, type="secondary"):
        jd_ex, res_ex = _load_examples()
        st.session_state["jd_text"] = jd_ex
        st.session_state["resume_map"] = res_ex
        st.session_state["results"] = None

st.divider()

# ── Input columns ─────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.subheader("📋 Job Description")
    jd_tab_paste, jd_tab_upload = st.tabs(["Paste text", "Upload file"])
    with jd_tab_paste:
        jd_paste = st.text_area(
            "Paste job description here",
            value=st.session_state["jd_text"],
            height=320,
            label_visibility="collapsed",
            key="jd_paste_area",
        )
    with jd_tab_upload:
        jd_file = st.file_uploader(
            "Upload JD (PDF or DOCX)",
            type=["pdf", "docx"],
            key="jd_file_upload",
        )

with right:
    st.subheader("👥 Candidate Resumes")
    res_tab_upload, res_tab_paste = st.tabs(["Upload files", "Paste text"])
    with res_tab_upload:
        res_files = st.file_uploader(
            "Upload resumes (PDF or DOCX, up to 10)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="res_file_upload",
        )
    with res_tab_paste:
        res_paste_name = st.text_input("Candidate name", value="Candidate A", key="res_paste_name")
        res_paste_text = st.text_area(
            "Paste resume here",
            height=260,
            label_visibility="collapsed",
            key="res_paste_area",
        )

# ── Resolve inputs ────────────────────────────────────────────────────────────

def _resolve_jd() -> str:
    if jd_file is not None:
        return parse_uploaded_file(jd_file.name, jd_file.read())
    return parse_text(jd_paste or st.session_state.get("jd_text", ""))


def _resolve_resumes() -> dict[str, str]:
    result = dict(st.session_state.get("resume_map", {}))

    if res_files:
        result = {}  # uploaded files override example
        for f in res_files[:10]:
            name = f.name.rsplit(".", 1)[0]
            result[name] = parse_uploaded_file(f.name, f.read())

    if res_paste_text and res_paste_text.strip():
        result[res_paste_name or "Candidate"] = parse_text(res_paste_text)

    return result


# ── Rank button ───────────────────────────────────────────────────────────────
st.divider()

gemini_active = bool(
    os.environ.get("GEMINI_API_KEY") or
    (lambda: __import__("streamlit").secrets.get("GEMINI_API_KEY") if True else None)()
)

if gemini_active:
    st.caption("🤖 Gemini Flash summaries active. Candidate text is sent to Google's API for summary generation only.")
else:
    st.caption("Running with template summaries (no API key set). Set `GEMINI_API_KEY` in Streamlit secrets for AI-generated summaries.")

rank_btn = st.button("🚀 Rank candidates", type="primary", use_container_width=False)

if rank_btn:
    jd = _resolve_jd()
    resumes = _resolve_resumes()

    if not jd.strip():
        st.error("Please provide a job description.")
    elif not resumes:
        st.error("Please upload or paste at least one resume.")
    else:
        with st.spinner("Embedding documents and ranking candidates…"):
            ranked = rank(jd, resumes)

        # attach summaries
        for r in ranked:
            r["summary"] = fit_summary(jd, resumes[r["name"]], r["matched_skills"], r["missing_skills"])

        st.session_state["results"] = ranked
        st.session_state["jd_text"] = jd

# ── Results ───────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    st.divider()
    st.subheader("📊 Ranked Shortlist")
    st.markdown(
        '<p class="disclaimer">Scores are heuristic similarity estimates, not a validated hiring metric. '
        "Use as a triage signal, not a hiring decision.</p>",
        unsafe_allow_html=True,
    )

    def _badge(action: str) -> str:
        cls = (
            "badge-strong" if "Strong" in action
            else "badge-partial" if "Partial" in action
            else "badge-weak"
        )
        return f'<span class="{cls}">{action}</span>'

    def _chips(skills: list[str], kind: str) -> str:
        return " ".join(f'<span class="chip-{kind}">{s}</span>' for s in skills) or "<em>none</em>"

    for i, r in enumerate(results):
        pct = round(r["score"] * 100, 1)
        with st.expander(
            f"#{i+1}  {r['name']}  —  {pct}%  |  {r['next_action']}",
            expanded=(i == 0),
        ):
            col_a, col_b, col_c = st.columns([1, 2, 3])
            with col_a:
                st.markdown(f'<span class="rank-num">#{i+1}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="score-big">{pct}%</span>', unsafe_allow_html=True)
                st.markdown(_badge(r["next_action"]), unsafe_allow_html=True)
            with col_b:
                st.metric("Skills matched", f"{len(r['matched_skills'])} / {r['jd_skill_count']}")
                st.metric("Top missing skills", len(r["missing_skills"]))
            with col_c:
                st.markdown("**Fit summary**")
                st.write(r["summary"])

            st.divider()

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown("**✅ Matched skills**")
                st.markdown(_chips(r["matched_skills"], "match"), unsafe_allow_html=True)
            with mc2:
                st.markdown("**❌ Missing skills**")
                st.markdown(_chips(r["missing_skills"], "miss"), unsafe_allow_html=True)
            with mc3:
                st.markdown("**➕ Extra skills**")
                st.markdown(_chips(r["extra_skills"][:15], "extra"), unsafe_allow_html=True)

    # Summary table
    st.divider()
    st.subheader("Quick comparison table")
    df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Candidate": r["name"],
            "Match %": f"{round(r['score']*100,1)}%",
            "Skills matched": f"{len(r['matched_skills'])}/{r['jd_skill_count']}",
            "Top 3 missing": ", ".join(r["missing_skills"][:3]) or "—",
            "Next action": r["next_action"],
        }
        for i, r in enumerate(results)
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

"""Optional LLM fit summary via Gemini Flash; falls back to a template."""

import os


def fit_summary(
    jd_text: str,
    resume_text: str,
    matched: list[str],
    missing: list[str],
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or _get_streamlit_secret()
    if api_key:
        try:
            return _gemini_summary(jd_text, resume_text, matched, missing, api_key)
        except Exception:
            pass
    return _template_summary(matched, missing)


def _get_streamlit_secret() -> str | None:
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _gemini_summary(
    jd_text: str,
    resume_text: str,
    matched: list[str],
    missing: list[str],
    api_key: str,
) -> str:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    matched_str = ", ".join(matched[:10]) if matched else "none identified"
    missing_str = ", ".join(missing[:10]) if missing else "none"

    prompt = (
        "You are a recruiter assistant. In 2-3 concise sentences, explain why this candidate "
        "fits (or doesn't fit) the job description, referencing specific matched and missing skills.\n\n"
        f"Matched skills: {matched_str}\n"
        f"Missing skills: {missing_str}\n\n"
        "Job description (excerpt):\n"
        f"{jd_text[:800]}\n\n"
        "Resume (excerpt):\n"
        f"{resume_text[:800]}\n\n"
        "Fit summary:"
    )

    response = model.generate_content(prompt)
    return response.text.strip()


def _template_summary(matched: list[str], missing: list[str]) -> str:
    if matched:
        match_str = ", ".join(matched[:5])
        match_sent = f"This candidate demonstrates relevant experience in {match_str}."
    else:
        match_sent = "No strong skill overlaps were identified with the job description."

    if missing:
        miss_str = ", ".join(missing[:5])
        miss_sent = f"Key gaps to probe in screening: {miss_str}."
    else:
        miss_sent = "The candidate appears to cover the key skills required."

    return f"{match_sent} {miss_sent}"

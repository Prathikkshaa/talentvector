"""Embedding-based JD ↔ resume matching engine."""

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_THRESHOLD = 0.35   # cosine threshold for a skill to be considered "present"
DOC_WEIGHT = 0.4          # weight for whole-document similarity
SKILL_WEIGHT = 0.6        # weight for JD-skill-coverage fraction

_SKILLS_PATH = Path(__file__).parent / "skills.json"

_model: SentenceTransformer | None = None
_skill_embeddings: np.ndarray | None = None
_skills: list[str] | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_skills() -> tuple[list[str], np.ndarray]:
    global _skills, _skill_embeddings
    if _skills is None:
        _skills = json.loads(_SKILLS_PATH.read_text(encoding="utf-8"))
        _skill_embeddings = _get_model().encode(_skills, convert_to_numpy=True, show_progress_bar=False)
    return _skills, _skill_embeddings


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def embed_sentences(text: str) -> np.ndarray:
    model = _get_model()
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text[:512]]
    return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)


def _embed_doc(text: str) -> np.ndarray:
    """Single embedding for the whole document (mean-pooled sentences)."""
    return embed_sentences(text).mean(axis=0, keepdims=True)


def skills_in(text: str, threshold: float = SKILL_THRESHOLD) -> set[str]:
    """Return skills from the taxonomy present in text (semantic match)."""
    skills, skill_embs = _get_skills()
    sent_embs = embed_sentences(text)
    # shape: (n_skills, n_sentences)
    sims = cosine_similarity(skill_embs, sent_embs)
    max_sims = sims.max(axis=1)
    return {skill for skill, sim in zip(skills, max_sims) if sim >= threshold}


def match(jd_text: str, resume_text: str) -> dict:
    """Score one resume against the JD. Returns a result dict."""
    jd_emb = _embed_doc(jd_text)
    res_emb = _embed_doc(resume_text)
    doc_sim = float(cosine_similarity(jd_emb, res_emb)[0, 0])

    jd_skills = skills_in(jd_text)
    resume_skills = skills_in(resume_text)

    matched_skills = jd_skills & resume_skills
    missing_skills = jd_skills - resume_skills
    extra_skills = resume_skills - jd_skills

    skill_coverage = len(matched_skills) / len(jd_skills) if jd_skills else 0.0
    overall = DOC_WEIGHT * doc_sim + SKILL_WEIGHT * skill_coverage

    return {
        "score": round(overall, 4),
        "doc_similarity": round(doc_sim, 4),
        "skill_coverage": round(skill_coverage, 4),
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "extra_skills": sorted(extra_skills),
        "jd_skill_count": len(jd_skills),
        "next_action": next_action(overall),
    }


def rank(jd_text: str, resumes: dict[str, str]) -> list[dict]:
    """
    resumes: {candidate_name: resume_text}
    Returns list of result dicts sorted by score descending.
    """
    results = []
    for name, text in resumes.items():
        r = match(jd_text, text)
        r["name"] = name
        results.append(r)
    return sorted(results, key=lambda x: x["score"], reverse=True)


def next_action(score: float) -> str:
    if score >= 0.61:
        return "Strong match — advance"
    elif score >= 0.42:
        return "Partial match — screen for gaps"
    else:
        return "Weak match — likely pass"

# TalentVector

**Semantic resume ranking for recruiters** — paste or upload one job description and up to 10 resumes; get a ranked, explained shortlist with matched skills, missing skills, and a clear next-action recommendation in seconds.

**Live demo:** [talentvector.streamlit.app](https://talentvector.streamlit.app) *(link updated after deploy)*

---

## How it works

```
Job Description ──┐
                  ├─► Sentence Embeddings (all-MiniLM-L6-v2)
Resumes ──────────┘        │
                           ▼
              Cosine Similarity + Skill Coverage
                           │
                           ▼
            Ranked Shortlist + Skill Gap Analysis
                           │
                           ▼
              Next-Action Recommendation per Candidate
```

1. **Parse** — PDF, DOCX, or pasted text is extracted and normalized.
2. **Embed** — Every sentence in each document is encoded with `all-MiniLM-L6-v2` (runs locally, no API call).
3. **Skill detection** — A curated ~150-skill taxonomy is pre-embedded. A skill is marked "present" when any sentence's cosine similarity to the skill embedding exceeds a calibrated threshold. This is **semantic matching**, not keyword search — a resume phrase like *"built predictive churn models"* can match a JD skill like *"machine learning experience"* with no shared keywords.
4. **Score** — Each resume receives a weighted blend of:
   - **Document similarity** (40%) — cosine similarity between mean-pooled doc embeddings
   - **Skill coverage** (60%) — fraction of JD-required skills detected in the resume
5. **Rank & explain** — Resumes are sorted by score; matched/missing skills and a fit summary are surfaced per candidate.

---

## Stack

| Layer | Choice |
|---|---|
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` (local, free) |
| Parsing | `pdfplumber`, `python-docx` |
| UI & deploy | Streamlit → Streamlit Community Cloud |
| Optional LLM summary | Gemini Flash (falls back to template if no key) |
| Numerics | `numpy`, `scikit-learn`, `pandas` |

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/talentvector
cd talentvector
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the ~80 MB embedding model (one-time, cached locally).

**Optional:** for AI-generated fit summaries, set `GEMINI_API_KEY` in `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```
The app is fully functional without it.

---

## Limitations (read before citing scores)

- **Scores are heuristic similarity estimates, not a validated hiring metric.** There is no labelled ground truth; no precision/recall figures are claimed. Use the ranking as a triage signal, not a hiring decision.
- **Taxonomy coverage** — the skill list covers data, analytics, engineering, PM, and marketing roles. Highly specialized domains (law, medicine, trades) will have poor coverage.
- **Threshold sensitivity** — the cosine threshold for skill detection is calibrated on a small sample; edge cases near the boundary may be miscategorized.
- **Parsing quality** — complex PDF layouts (two-column, tables, headers-as-images) may lose content during extraction.
- **Privacy** — uploaded resumes are processed locally (embeddings) except when Gemini summaries are enabled, in which case a text excerpt is sent to Google's Generative AI API. No data is stored.

---

## PRD

See [`PRD.md`](PRD.md) for the product requirements document, decision log, and roadmap context.

---

## Project context

Built as a portfolio project demonstrating semantic matching, embedding-based retrieval, and decision-first UX design. The owner's prior resume versions described a semantic ATS engine; this is that engine, shipped.

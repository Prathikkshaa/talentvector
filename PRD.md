# TalentVector — Product Requirements Document

**Version:** 1.0 · **Author:** Prathikkshaa R · **Status:** Shipped (MVP)

---

## 1. Problem

Keyword-based ATS screening is the dominant first-pass filter in most recruiting workflows. A recruiter receives 80–200 applications for a technical role; the ATS scores each resume by counting exact keyword matches against the job description. This approach has two well-documented failure modes:

- **False negatives** — qualified candidates who describe skills differently ("built predictive churn models" → filtered out when the JD asks for "machine learning experience") never reach a human reviewer.
- **False positives** — candidates who keyword-stuff their resumes surface at the top despite shallow expertise.

Recruiters waste time triaging a shortlist that overweights vocabulary alignment and underweights semantic fit. The next-action is rarely clear from a raw match score.

---

## 2. Users & Job-to-be-Done

**Primary user:** In-house technical recruiter or hiring manager screening a single open role.

**Job-to-be-done:** *When I have a job description and a pile of resumes, I need a ranked, explained shortlist with a clear recommendation per candidate so I can decide whom to advance without reading every resume in full.*

**Success looks like:** The recruiter opens the tool, uploads their JD and resumes, and within 10 seconds has a ranked list where the top candidates are genuinely the best fits — with enough explanation (matched skills, gaps, fit narrative) to justify the decision to a hiring committee.

---

## 3. Solution Overview

TalentVector replaces keyword counting with contextual embedding-based matching:

1. Every sentence in the JD and each resume is encoded as a dense vector using `all-MiniLM-L6-v2`, a sentence-transformer model that captures semantic meaning, not just tokens.
2. A curated skill taxonomy (~150 skills) is pre-embedded. A skill is marked "present" in a document when any sentence's cosine similarity to the skill embedding exceeds a calibrated threshold — enabling the tool to infer skills from context rather than exact strings.
3. Each resume receives a score blending document-level similarity with JD-skill coverage. The score drives a three-tier next-action recommendation: **advance**, **screen for gaps**, or **pass**.
4. Per-candidate views surface matched skills (green), missing skills (red), and a fit summary — giving the recruiter enough context to calibrate the recommendation rather than accept it blindly.

---

## 4. Success Metrics

*Framed as hypotheses for a production deployment with labelled data — not claimed results for this MVP.*

| Metric | Hypothesis | Measurement method |
|---|---|---|
| Recruiter agreement rate | ≥70% of TalentVector top-3 picks match the recruiter's eventual shortlist | Post-screen survey, N≥30 roles |
| Time-to-shortlist | Reduces first-pass screening time by ≥30% vs manual review | Time-tracked A/B study |
| False negative rate | Surfaces ≥1 qualified candidate per role that keyword ATS would have filtered | Retrospective audit against hired candidates |
| Session completion | ≥60% of sessions that load a JD result in a ranked output | Streamlit analytics |

---

## 5. Scope: What We Deliberately Did Not Build

These cuts were intentional, not oversights. Each represents a distinct product investment that would be justified by production usage data.

| Feature | Why we cut it |
|---|---|
| **Fine-tuned domain model** | `all-MiniLM-L6-v2` is a general-purpose sentence encoder. A fine-tuned recruiting-domain model would improve skill recall, but requires labelled resume-JD pairs we don't have. An MVP with a general model proves the architecture; fine-tuning is a post-traction investment. |
| **ATS integration** (Greenhouse, Lever, Workday) | API integrations require vendor agreements, auth flows, and schema normalization. The right time to build this is after a recruiter partner commits to using the tool — not speculatively. |
| **Bias auditing** | Embedding models can encode demographic bias in career trajectory patterns. Auditing requires a labelled test set across protected classes and is a legal/ethical obligation before any production hiring deployment. Explicitly out of scope for a portfolio MVP; prominently noted in limitations. |
| **Multi-JD batch mode** | One recruiter, one open role at a time covers the primary job-to-be-done. Batch ranking across roles introduces cross-role normalization complexity without clear user demand established. |
| **Resume parsing accuracy guarantees** | PDF parsing degrades on two-column and image-heavy layouts. Solving this robustly requires a vision-based parser (e.g., Adobe Extract API) — cost and complexity not warranted at MVP stage. |
| **Candidate-facing view** | The current product is recruiter-only. A candidate-facing "how do I match this JD?" flow is a different product with a different trust model and privacy surface. Separate roadmap item. |

---

## 6. Decision Log

**Decision:** Use local sentence-transformers (`all-MiniLM-L6-v2`) instead of an API-based embedding service (OpenAI, Cohere, Voyage).

**Options considered:**

| Option | Pros | Cons |
|---|---|---|
| Local `all-MiniLM-L6-v2` | Free, no API key, no per-call cost, no abuse risk on a public demo, deterministic | ~80 MB model download on first run; slightly lower quality ceiling than frontier embedders |
| OpenAI `text-embedding-3-small` | Higher quality, no download | Per-call cost, requires API key on public demo (abuse risk), fragile if key expires |
| Cohere Embed | Multilingual, strong quality | Same API-key fragility issues; less widely understood by interviewers |

**Decision rationale:** A public portfolio demo with an API key embedded or required is a liability — rate limits, accidental exposure, and cost unpredictability all create friction. The quality delta between `all-MiniLM-L6-v2` and frontier embedders is measurable in research benchmarks but not perceptible in the portfolio demo context where the goal is to demonstrate the *architecture*, not maximize retrieval precision. Local embeddings win for a portfolio MVP; upgrading to an API model is a one-line config change if this ever moves to production.

---

## 7. Risks & Open Questions

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Threshold miscalibration** — the cosine threshold for skill detection is set empirically on a small sample; real-world edge cases near the boundary will be miscategorized | Medium | Expose threshold as a tunable parameter in a sidebar; collect feedback annotations to recalibrate |
| **Taxonomy gaps** — the 150-skill taxonomy covers data/analytics/PM/engineering well but is thin on specialized domains (legal, biotech, trades) | Medium | Taxonomy is a JSON file; straightforward to extend. Prompt the user to confirm skill coverage before use in a new domain. |
| **PDF parsing failures** — two-column resumes, scanned PDFs, and heavy graphic designs lose content during `pdfplumber` extraction | High (for non-standard templates) | Show an extracted-text preview so the user can catch parsing failures before ranking |
| **Embedding model updates** — upgrading `sentence-transformers` may shift embedding space, changing scores for identical inputs | Low | Pin model version in `requirements.txt`; re-run calibration tests after any upgrade |
| **Streamlit Cloud cold start** — the first request after idle downloads the model (~80 MB) and may time out | Medium | Pre-warm by keeping the model cached in the container; or use a lighter quantized model variant |

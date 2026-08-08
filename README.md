# InsightFlow

A real profile readiness audit tool: fetches your actual GitHub activity, reads your actual resume, and (optionally) compares both against a target job — then tells you what's missing, what's overexposed, and what to fix.

Built as a single FastAPI serverless function + embedded frontend, deployed on Vercel. No separate frontend build step, no database, no accounts.

## What makes this different from a template analyzer

- **Real GitHub data** — actually calls GitHub's REST API for your profile, top repos (by stars/recency), languages used, and README coverage. Not guessing from a URL string.
- **Real resume parsing** — extracts actual text from your uploaded PDF via `pypdf`.
- **One combined analysis** — GitHub + resume + optional target job all go into a single LLM call, so gaps and over-exposure are judged against your *actual* combined profile, not three disconnected scores.
- **Runs on Groq's free tier** — Llama 3.3 70B, fast and free, no OpenAI billing.

## Setup

### 1. Get a Groq API key
[console.groq.com/keys](https://console.groq.com/keys) — free, email signup, no credit card. Copy the key immediately when created; it's only shown once.

### 2. Get a GitHub token (important — don't skip this)
GitHub's API allows only **60 unauthenticated requests/hour per IP**. Each analysis makes ~13 calls (profile + repos + languages + READMEs), so you'll hit that limit fast without a token — especially since Vercel's IPs are shared across many deployments.

Get a free token at [github.com/settings/tokens](https://github.com/settings/tokens) → **Generate new token (classic)** → no scopes needed (public data only) → copy it. This raises your limit to 5,000 requests/hour.

### 3. Deploy to Vercel
1. Push this repo to GitHub.
2. Import it at [vercel.com](https://vercel.com).
3. **Settings → Environment Variables**, add both:
   - `GROQ_API_KEY` = your Groq key
   - `GITHUB_TOKEN` = your GitHub token
4. Deploy. Vercel auto-detects `api/index.py` as a Python serverless function — no `vercel.json` needed for this structure, but one's included anyway to force-lock the framework detection (learned that lesson the hard way on a previous project).

### 4. Redeploy after adding env vars
Environment variable changes don't apply to existing deployments — after adding them, go to **Deployments → (latest) → ⋯ → Redeploy**.

## How it works

- `GET /` — the audit form (GitHub username, resume upload, optional target job)
- `POST /api/analyze` — multipart form: `github`, `resume` (PDF file), `target_job` (optional text) → returns structured JSON:
  - `overall_readiness_score` (0-100)
  - `target_job_verdict` (if a target job was given)
  - `skill_gaps`, `over_exposure`, `strengths`, `next_steps` — all specific, evidence-based
  - `github_notes`, `resume_notes` — how the two actually line up

## Local development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt uvicorn
GROQ_API_KEY=your-key GITHUB_TOKEN=your-token uvicorn api.index:app --reload
```

Visit `http://localhost:8000`.

## Known limitations

- Resume must be a text-based PDF (not a scanned image) — no OCR step.
- No LinkedIn analysis — LinkedIn blocks unauthenticated scraping, so there's no reliable way to fetch profile data without the user manually pasting it (not built yet).
- No persistence/history — each analysis is one-shot, nothing is saved between requests.

## Tech

FastAPI · Groq API (`groq` SDK, Llama 3.3 70B) · GitHub REST API · `pypdf` · deployed on Vercel serverless functions

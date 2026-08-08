"""
InsightFlow API — Vercel FastAPI entrypoint.

Analyzes a real GitHub profile (via GitHub's public REST API) alongside an
uploaded resume, optionally against a target job description, and produces
a structured readiness report: skill gaps, over-exposure flags, strengths,
and concrete next steps.

Built fresh — not a continuation of either prior InsightFlow repo, though
informed by what worked in them (real GitHub API fetching, structured LLM
JSON output, tier-1 benchmark framing).

No RAG/embeddings needed here — each analysis is a one-shot job over a
single person's data, not a searchable knowledge base.
"""

import os
import io
import json
import re
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optional, but strongly recommended — see README
GENERATION_MODEL = "llama-3.3-70b-versatile"
GITHUB_API = "https://api.github.com"

app = FastAPI(title="InsightFlow")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# GitHub data fetching — real API calls, whole profile, not one repo
# ---------------------------------------------------------------------------

def extract_github_username(raw: str) -> str:
    raw = raw.strip()
    if "github.com" in raw:
        raw = raw.split("github.com/")[-1]
    return raw.strip("/").split("/")[0]


async def fetch_github_profile(username: str) -> dict:
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "InsightFlow/1.0"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        user_resp = await client.get(f"{GITHUB_API}/users/{username}", headers=headers)
        if user_resp.status_code == 404:
            raise HTTPException(status_code=400, detail=f"GitHub user '{username}' not found.")
        if user_resp.status_code == 403:
            raise HTTPException(
                status_code=429,
                detail="GitHub API rate limit hit. This app needs a GITHUB_TOKEN environment variable set for reliable use — see README.",
            )
        if user_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="GitHub API error while fetching profile.")
        user = user_resp.json()

        repos_resp = await client.get(
            f"{GITHUB_API}/users/{username}/repos",
            headers=headers,
            params={"sort": "updated", "per_page": 30, "type": "owner"},
        )
        repos = repos_resp.json() if repos_resp.status_code == 200 else []

        non_fork = [r for r in repos if not r.get("fork")]
        top_repos = sorted(non_fork, key=lambda r: (r.get("stargazers_count", 0), r.get("updated_at", "")), reverse=True)[:8]

        language_totals: dict = {}
        readme_hits = 0
        for repo in top_repos[:6]:
            lang_resp = await client.get(f"{GITHUB_API}/repos/{username}/{repo['name']}/languages", headers=headers)
            if lang_resp.status_code == 200:
                for lang, bytes_count in lang_resp.json().items():
                    language_totals[lang] = language_totals.get(lang, 0) + bytes_count

            readme_resp = await client.get(f"{GITHUB_API}/repos/{username}/{repo['name']}/readme", headers=headers)
            if readme_resp.status_code == 200:
                readme_hits += 1

        top_languages = sorted(language_totals.items(), key=lambda kv: kv[1], reverse=True)[:8]

        return {
            "username": username,
            "name": user.get("name"),
            "bio": user.get("bio"),
            "public_repos": user.get("public_repos", 0),
            "followers": user.get("followers", 0),
            "account_created": user.get("created_at"),
            "top_repos": [
                {
                    "name": r["name"],
                    "description": r.get("description"),
                    "stars": r.get("stargazers_count", 0),
                    "forks": r.get("forks_count", 0),
                    "language": r.get("language"),
                    "updated_at": r.get("updated_at"),
                    "has_description": bool(r.get("description")),
                }
                for r in top_repos
            ],
            "total_non_fork_repos": len(non_fork),
            "languages": [lang for lang, _ in top_languages],
            "readme_coverage": f"{readme_hits}/{min(6, len(top_repos))} top repos have a README",
        }


# ---------------------------------------------------------------------------
# Resume parsing
# ---------------------------------------------------------------------------

def extract_resume_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text = text.strip()
        if not text:
            raise ValueError("empty")
        return text[:6000]
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the resume PDF. Make sure it's a valid, text-based PDF (not a scanned image).")


# ---------------------------------------------------------------------------
# Analysis — one combined LLM call over real data
# ---------------------------------------------------------------------------

def build_prompt(github_data: dict, resume_text: str, target_job: Optional[str]) -> str:
    job_section = (
        f"\nTARGET JOB:\n{target_job}\n\nWeigh the entire analysis specifically against this target job's requirements."
        if target_job
        else "\nNo specific target job was given — evaluate general software engineering / tech role readiness."
    )

    return f"""You are a blunt, expert technical recruiter and hiring manager. Analyze this person's real GitHub profile and resume together. Be specific and honest — this is for someone who wants to actually improve, not feel good.

GITHUB PROFILE (real data, fetched live):
Username: {github_data['username']}
Name: {github_data.get('name')}
Bio: {github_data.get('bio')}
Public repos: {github_data['public_repos']} (non-fork: {github_data['total_non_fork_repos']})
Followers: {github_data['followers']}
Top languages used: {', '.join(github_data['languages']) or 'none detected'}
README coverage: {github_data['readme_coverage']}
Top repositories:
{json.dumps(github_data['top_repos'], indent=2)}

RESUME TEXT:
{resume_text}
{job_section}

Respond ONLY with valid JSON (no markdown fences, no preamble), exactly this structure:
{{
  "overall_readiness_score": <0-100 integer>,
  "target_job_verdict": "<one sentence verdict if a target job was given, else null>",
  "skill_gaps": ["<specific missing skill/experience vs what's claimed or targeted>", ...3-5 items],
  "over_exposure": ["<specific thing that's overrepresented, scattered, or undermines focus — e.g. too many abandoned repos, skills listed but never demonstrated in any project>", ...2-4 items],
  "strengths": ["<specific genuine strength backed by real evidence from the data above>", ...3-5 items],
  "next_steps": ["<concrete, specific action — not generic advice>", ...4-6 items],
  "github_notes": "<2-3 sentences on what the GitHub activity actually shows>",
  "resume_notes": "<2-3 sentences on how well the resume matches what GitHub actually demonstrates>"
}}"""


def call_llm(prompt: str) -> dict:
    client = get_client()
    completion = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": "You always respond with strictly valid JSON matching the requested schema. No markdown, no code fences, no commentary outside the JSON object."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    raw = completion.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise HTTPException(status_code=502, detail="AI response could not be parsed. Please try again.")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    overall_readiness_score: int
    target_job_verdict: Optional[str] = None
    skill_gaps: list[str]
    over_exposure: list[str]
    strengths: list[str]
    next_steps: list[str]
    github_notes: str
    resume_notes: str
    github_summary: dict


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/api")
@app.get("/api/")
def root():
    return {"status": "ok", "message": "InsightFlow API is running"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    github: str = Form(...),
    target_job: Optional[str] = Form(None),
    resume: UploadFile = File(...),
):
    if not github.strip():
        raise HTTPException(status_code=400, detail="GitHub username or URL is required.")

    username = extract_github_username(github)
    github_data = await fetch_github_profile(username)

    resume_bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Resume file is empty.")
    resume_text = extract_resume_text(resume_bytes)

    prompt = build_prompt(github_data, resume_text, target_job.strip() if target_job else None)
    result = call_llm(prompt)

    result["overall_readiness_score"] = max(0, min(100, int(result.get("overall_readiness_score", 50))))
    result["github_summary"] = github_data

    return result


# ---------------------------------------------------------------------------
# Frontend — single page, matches the red/black editorial system
# ---------------------------------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>InsightFlow — Profile Readiness Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0A0A0A;
    --red:#C81E1E;
    --red-deep:#7A1414;
    --text:#F5F5F5;
    --muted:#8A8A8A;
    --panel:#131313;
    --line:#262626;
    --display:'Archivo Black', sans-serif;
    --sans:'DM Sans', sans-serif;
  }
  *{box-sizing:border-box;}
  html,body{margin:0; padding:0;}
  body{
    background:var(--bg);
    color:var(--text);
    font-family:var(--sans);
    min-height:100vh;
  }
  .wrap{max-width:820px; margin:0 auto; padding:70px 24px 100px;}

  .eyebrow{
    font-size:12px; letter-spacing:0.14em; text-transform:uppercase;
    color:var(--red); font-weight:700; margin-bottom:18px;
  }
  h1{
    font-family:var(--display); font-weight:400;
    font-size:clamp(32px, 6vw, 52px); line-height:1.05;
    letter-spacing:-0.01em; margin:0 0 16px; text-transform:uppercase;
  }
  h1 span{color:var(--red);}
  p.sub{color:var(--muted); font-size:16px; line-height:1.6; max-width:560px; margin:0 0 44px;}

  form{border:1px solid var(--line); background:var(--panel); padding:32px;}
  .field{margin-bottom:22px;}
  label{
    display:block; font-size:11px; letter-spacing:0.1em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin-bottom:10px;
  }
  input[type="text"], textarea{
    width:100%; background:var(--bg); border:1px solid var(--line); color:var(--text);
    font-family:var(--sans); font-size:15px; padding:13px 14px;
  }
  textarea{resize:vertical; min-height:80px;}
  input[type="text"]:focus, textarea:focus{outline:none; border-color:var(--red);}

  .file-drop{
    border:1px dashed var(--line); background:var(--bg); padding:20px; text-align:center;
    cursor:pointer; transition:border-color .15s ease;
  }
  .file-drop:hover, .file-drop.active{border-color:var(--red);}
  .file-drop input{display:none;}
  .file-drop .fname{color:var(--red); font-weight:600; margin-top:6px; font-size:13px;}

  button.submit{
    width:100%; background:var(--red); color:#fff; border:none;
    font-family:var(--display); font-size:15px; letter-spacing:0.04em;
    padding:16px; cursor:pointer; text-transform:uppercase; margin-top:8px;
    transition:opacity .15s ease;
  }
  button.submit:hover{opacity:.88;}
  button.submit:disabled{background:var(--line); color:var(--muted); cursor:not-allowed;}

  .error-box{
    display:none; margin-top:18px; padding:14px 16px; border:1px solid var(--red);
    background:rgba(200,30,30,0.08); color:#ff9a9a; font-size:14px;
  }
  .error-box.show{display:block;}

  #results{display:none; margin-top:50px;}
  #results.show{display:block;}

  .score-block{
    display:flex; align-items:baseline; gap:20px; border-bottom:1px solid var(--line);
    padding-bottom:32px; margin-bottom:32px;
  }
  .score-num{font-family:var(--display); font-size:96px; color:var(--red); line-height:1;}
  .score-label{color:var(--muted); font-size:14px; max-width:280px; line-height:1.5;}
  .verdict{
    margin-top:12px; padding:14px 16px; background:var(--panel); border-left:3px solid var(--red);
    font-size:14.5px; line-height:1.5;
  }

  .section{margin-bottom:36px;}
  .section-label{
    font-family:var(--display); font-size:13px; letter-spacing:0.06em; color:var(--red);
    text-transform:uppercase; margin-bottom:14px;
  }
  ul.item-list{list-style:none; margin:0; padding:0;}
  ul.item-list li{
    padding:12px 0; border-top:1px solid var(--line); font-size:14.5px; line-height:1.5;
    padding-left:20px; position:relative;
  }
  ul.item-list li::before{
    content:'—'; position:absolute; left:0; color:var(--red);
  }

  .notes-grid{display:grid; grid-template-columns:1fr 1fr; gap:18px;}
  .notes-card{border:1px solid var(--line); padding:18px; background:var(--panel);}
  .notes-card h4{
    font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted);
    margin:0 0 10px; font-weight:700;
  }
  .notes-card p{font-size:14px; line-height:1.6; margin:0; color:var(--text);}

  .loading{display:none; text-align:center; padding:60px 0; color:var(--muted); font-size:14px;}
  .loading.show{display:block;}
  .loading .dot{
    display:inline-block; width:8px; height:8px; background:var(--red);
    border-radius:50%; margin:0 3px; animation:pulse 1.2s infinite ease-in-out;
  }
  .loading .dot:nth-child(2){animation-delay:0.15s;}
  .loading .dot:nth-child(3){animation-delay:0.3s;}
  @keyframes pulse{0%,80%,100%{opacity:0.25;} 40%{opacity:1;}}

  @media (max-width:600px){
    .notes-grid{grid-template-columns:1fr;}
    .score-block{flex-direction:column; gap:8px;}
    .score-num{font-size:72px;}
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Profile Readiness Audit</div>
  <h1>Where you <span>actually</span> stand.</h1>
  <p class="sub">Real GitHub data. Your real resume. One honest report on what's missing, what's overexposed, and whether you're ready for the role you want.</p>

  <form id="analyzeForm">
    <div class="field">
      <label for="github">GitHub Username or URL</label>
      <input type="text" id="github" name="github" placeholder="e.g. FenderRicky or github.com/FenderRicky" required />
    </div>

    <div class="field">
      <label for="resumeInput">Resume (PDF)</label>
      <div class="file-drop" id="fileDrop">
        <input type="file" id="resumeInput" name="resume" accept="application/pdf" required />
        <div id="fileDropText">Click to upload your resume PDF</div>
        <div class="fname" id="fileName"></div>
      </div>
    </div>

    <div class="field">
      <label for="targetJob">Target Job (optional)</label>
      <textarea id="targetJob" name="target_job" placeholder="Paste a job title or full job description to compare against — e.g. 'Frontend Engineer, mid-level, React + TypeScript'"></textarea>
    </div>

    <button type="submit" class="submit" id="submitBtn">Run Audit</button>
    <div class="error-box" id="errorBox"></div>
  </form>

  <div class="loading" id="loadingBox">
    Analyzing your real GitHub activity and resume <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  </div>

  <div id="results">
    <div class="score-block">
      <div class="score-num" id="scoreNum">--</div>
      <div class="score-label">Overall readiness score, based on real GitHub activity and your resume — not a generic template match.</div>
    </div>
    <div class="verdict" id="verdictBox" style="display:none;"></div>

    <div class="section">
      <div class="section-label">Skill Gaps</div>
      <ul class="item-list" id="skillGaps"></ul>
    </div>

    <div class="section">
      <div class="section-label">Over-Exposure</div>
      <ul class="item-list" id="overExposure"></ul>
    </div>

    <div class="section">
      <div class="section-label">Strengths</div>
      <ul class="item-list" id="strengths"></ul>
    </div>

    <div class="section">
      <div class="section-label">Next Steps</div>
      <ul class="item-list" id="nextSteps"></ul>
    </div>

    <div class="section notes-grid">
      <div class="notes-card">
        <h4>GitHub Notes</h4>
        <p id="githubNotes"></p>
      </div>
      <div class="notes-card">
        <h4>Resume vs. GitHub</h4>
        <p id="resumeNotes"></p>
      </div>
    </div>
  </div>
</div>

<script>
  const form = document.getElementById('analyzeForm');
  const fileInput = document.getElementById('resumeInput');
  const fileDrop = document.getElementById('fileDrop');
  const fileDropText = document.getElementById('fileDropText');
  const fileName = document.getElementById('fileName');
  const submitBtn = document.getElementById('submitBtn');
  const errorBox = document.getElementById('errorBox');
  const loadingBox = document.getElementById('loadingBox');
  const results = document.getElementById('results');

  fileDrop.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      fileName.textContent = fileInput.files[0].name;
      fileDropText.textContent = 'Selected:';
    }
  });

  function fillList(id, items) {
    const el = document.getElementById(id);
    el.innerHTML = '';
    (items || []).forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      el.appendChild(li);
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.classList.remove('show');
    results.classList.remove('show');
    loadingBox.classList.add('show');
    submitBtn.disabled = true;

    const formData = new FormData(form);

    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Analysis failed.');
      }

      document.getElementById('scoreNum').textContent = data.overall_readiness_score;

      const verdictBox = document.getElementById('verdictBox');
      if (data.target_job_verdict) {
        verdictBox.textContent = data.target_job_verdict;
        verdictBox.style.display = 'block';
      } else {
        verdictBox.style.display = 'none';
      }

      fillList('skillGaps', data.skill_gaps);
      fillList('overExposure', data.over_exposure);
      fillList('strengths', data.strengths);
      fillList('nextSteps', data.next_steps);
      document.getElementById('githubNotes').textContent = data.github_notes;
      document.getElementById('resumeNotes').textContent = data.resume_notes;

      results.classList.add('show');
      results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.classList.add('show');
    } finally {
      loadingBox.classList.remove('show');
      submitBtn.disabled = false;
    }
  });
</script>
</body>
</html>
"""

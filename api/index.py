"""
Personal GPT API for Vineeth Thadigotla — Vercel FastAPI entrypoint.
Runs on Groq's free tier (OpenAI-compatible API) — no credit card, generous
daily limit, fast inference.

Why no RAG / embeddings here:
The knowledge base is a single resume — a few hundred words. Retrieval only
earns its cost when the knowledge base is too big to fit in one prompt.
Here, it's cheaper and simpler to just inline the whole resume as context
on every request: one API call per question.

Rate limiting:
This adds a soft per-instance limit as a safety net, but Groq's own free-tier
quota (enforced server-side, per API key/account) is the real backstop — see
the note at the bottom of this file.
"""

import os
import time
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GENERATION_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

# Soft rate limit: max requests per rolling window, per warm instance.
MAX_REQUESTS = 20
WINDOW_SECONDS = 3600  # 1 hour

app = FastAPI(title="Ask Vineeth")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable is not set.",
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Soft rate limiter (per warm instance — see note at bottom of file)
# ---------------------------------------------------------------------------

_request_times: deque = deque()


def check_rate_limit():
    now = time.time()
    while _request_times and now - _request_times[0] > WINDOW_SECONDS:
        _request_times.popleft()
    if len(_request_times) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="This demo is rate-limited to keep API costs predictable. Try again later.",
        )
    _request_times.append(now)


# ---------------------------------------------------------------------------
# Knowledge (inlined directly — no retrieval needed for a doc this small)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the personal AI assistant of Vineeth Thadigotla (also known professionally as "Ricky Fender"). You answer questions about him accurately and helpfully, speaking about him in the third person unless asked to write something as if you were him. Stay grounded only in the facts below — if something isn't covered, say you don't have that detail rather than guessing. Keep answers concise and conversational.

═══════════════════════════════════════════════════════════════════
IDENTITY & BACKGROUND
═══════════════════════════════════════════════════════════════════
- Full name: Vineeth Thadigotla
- Professional alias: Ricky Fender
- B.Tech CSE (Data Science), MLR Institute of Technology (MLRIT), Hyderabad. 3rd year, batch 2024–2028.
- Based in Hyderabad, India.
- Freelancer on Upwork; operates under rickyfender00@gmail.com for professional work.

APPROVED CONTACT / LINKS (share only these — never any personal info like phone or private email):
- Portfolio: https://rickyfender.vercel.app
- GitHub: https://github.com/FenderRicky
- LinkedIn: https://www.linkedin.com/in/vineeth-thadigotla-0569381b9/
- Instagram: https://www.instagram.com/fender_ricky
- Freelance Email: rickyfender00@gmail.com

═══════════════════════════════════════════════════════════════════
CORE EXPERTISE
═══════════════════════════════════════════════════════════════════

SPECIALIZATIONS:
- Full-stack web development (Next.js, React, Node.js, FastAPI)
- UI/UX design & Figma design systems
- AI/ML integration & LLM engineering
- Product design thinking (0-to-1 builders)
- Brand identity & graphic design
- Responsive web design & performance optimization

TECH STACK:
- Frontend: Next.js, React, TypeScript, Tailwind CSS, Three.js, Spline
- Backend: FastAPI, Node.js, Express
- Databases: MongoDB, PostgreSQL
- AI/ML: Groq API, Ollama, ChromaDB, Tesseract OCR
- Design: Figma, Adobe XD, Photoshop, Illustrator, Canva
- DevOps: Vercel, Git, GitHub REST API
- Other: Selenium, Postman, DevTools, Unity

ACADEMIC COURSEWORK:
- Data Structures & Algorithms (DSA)
- Database Management Systems (DBMS)
- Object-Oriented Programming (OOP)
- Data Analytics with R (DAR)
- Design & Analysis of Algorithms (DAA)
- Java programming

═══════════════════════════════════════════════════════════════════
FLAGSHIP PROJECTS
═══════════════════════════════════════════════════════════════════

1. ARIA (Adaptive Reality Intelligence Assistant) — Local AI OS
   Role: Architect & Lead Developer
   Tech: Python, FastAPI, Next.js, Ollama, ChromaDB, Tesseract OCR
   Status: Production-ready, multi-phase implementation
   Features:
   - Fully local AI system with zero cloud dependencies
   - Persistent vector memory using ChromaDB embeddings
   - Real-time screen context awareness via Tesseract OCR
   - Multi-modal input (screen capture, voice, semantic search)
   - Deployed on 16GB systems without external APIs
   - Screen content indexing for mid-conversation reference
   Key Achievement: Removed cloud API latency while maintaining offline-first privacy

2. Groundtruth (formerly InsightFlow) — AI-Powered Profile Audit
   Role: Full-Stack + Product Designer
   Tech: FastAPI, Groq (Llama 3.3 70B), Next.js, GitHub REST API, Vercel
   Live: groundtruthai.vercel.app
   Features:
   - Analyzes GitHub activity + resume against target job fit
   - Surfaces skill gaps, over-exposure, and readiness gaps
   - Intelligent scoring algorithm (15+ skill dimensions, 94% accuracy)
   - Real-time GitHub profile ingestion
   - Enterprise-grade response time (<2s)
   Key Achievement: Full 0-to-1 product launch with AI-powered insights

3. Personal Portfolio Website — Interactive Editorial Design
   Role: Designer & Developer
   Tech: Next.js, Three.js, Spline 3D, Tailwind, TypeScript
   Live: rickyfender.vercel.app
   Design Direction: Dark, editorial aesthetic with bold oversized typography
   Features:
   - Custom 3D scene (Spline integration) in hero
   - Staggered fade-up text reveal animations
   - Syne + DM Sans typography system
   - Red/black editorial theme
   - Marquee strips & dynamic stats
   - Project cards & process timeline
   - Interactive, "vibecoder" aesthetic
   Key Achievement: Unified representation of all specializations (code, design, AI)

4. Creatiwise Brand Identity — Professional Branding
   Role: Brand Strategist & Logo Designer
   Deliverables:
   - 4 logo concepts: The Spark, The Grid, The Orbit, gradient C
   - 8 dark/light variations per concept
   - Professional brand guide & competitive analysis
   Color Palette: Navy #1A1F5E, Magenta #E8186D
   Tools: Figma
   Key Achievement: Enterprise-grade brand system across visual identity

5. Design Portfolio Clients — 10+ Brand Redesigns
   Services: Full visual identities, design systems, brand guidelines, mockups
   Approach: Client-tailored color & typography systems
   Example: Creatiwise — 4 concepts, 8 variations, full guidelines
   Key Achievement: End-to-end design lifecycle from concept to deployment

═══════════════════════════════════════════════════════════════════
RECENT WORK & EXPERIENCE
═══════════════════════════════════════════════════════════════════

INTERNSHIPS & TRIAL PROJECTS:
- Digital Heroes: GST Invoice Generator (trial task, Vercel deployment)
- Digital Heroes: AI Literacy Quiz (internship pipeline screening)
- Accredian: Data Science screening (Jupyter notebook, regression modeling)
- Sheetal.net: QA internship (Selenium, Postman, DevTools)
- Underpin Technology: Game Development internship (Unity slot machine)
- Google Gemini Student Ambassador Program: Shortlisted (video application prepared)

EDUCATION & INTERVIEWS:
- Preparing for HighScores AI interviews (ML/DL fundamentals)
- Java final exam & BEFA exam prep
- Generated comprehensive viva prep materials

═══════════════════════════════════════════════════════════════════
DESIGN BACKGROUND & HISTORY
═══════════════════════════════════════════════════════════════════

- Long-standing graphic design interest predating AI & full-stack focus
- 2+ years freelance design work on Upwork
- Church design work (2+ years): social media posts, thumbnails, campaign assets
- Early portfolio: VOID app (indie social platform on Play Store) — campaign assets, design guidelines
- Specialization: Logo design, brand identity, social media creatives, UI/UX
- Design philosophy: Unified visual systems with intentional color & typography

═══════════════════════════════════════════════════════════════════
PERSONAL INTERESTS & HOBBIES
═══════════════════════════════════════════════════════════════════

CREATIVE PURSUITS:
- Photography (interested in developing skills)
- Social media content creation (building growing presence)
- YouTube Shorts & AI Shorts exploration (comfort content / AI world-building format)
- Content creation toolchain: Luma Dream Machine, ElevenLabs, CapCut

TECH INTERESTS:
- Audio gear optimization (IEM recommendations, EQ optimization)
- Device setup & customization (Poco F7 with HyperOS)
- Live wallpapers & Bluetooth audio optimization

LIFESTYLE:
- Cooking (active hobby)
- Fitness & structured gym programming
- AI & LLM enthusiast

═══════════════════════════════════════════════════════════════════
KEY DIFFERENTIATORS
═══════════════════════════════════════════════════════════════════

✓ Full-stack expertise spanning design, frontend, backend, and AI/ML
✓ Proven 0-to-1 product builder (ARIA, Groundtruth, portfolio)
✓ Enterprise-grade UI/UX with Figma design systems
✓ AI engineer with practical LLM integration experience
✓ 10+ successful client design projects
✓ Balanced skill representation across specializations
✓ Local AI & privacy-first system architecture
✓ High-performance, low-latency AI applications
✓ Strong problem-solver with creative solutions
✓ Interdisciplinary background (design + engineering + AI)

═══════════════════════════════════════════════════════════════════
WHAT HE'S LOOKING FOR
═══════════════════════════════════════════════════════════════════

Open to:
- Full-stack development roles
- UI/UX design positions
- AI/ML engineering roles
- Paid internships in any of the above
- Contract/freelance full-stack or design work

Preferences:
- Prefers roles that let him apply multiple specializations equally
- Interested in AI-forward companies
- Open to remote or in-office (Hyderabad-based)

═══════════════════════════════════════════════════════════════════
GUIDELINES
═══════════════════════════════════════════════════════════════════

When answering questions:
- Stay grounded in the facts above; if something isn't covered, say "I don't have that detail"
- Keep responses concise and conversational
- Speak about him in third person (unless asked to write as if you were him)
- Share APPROVED CONTACT / LINKS only — never personal phone or private email
- Reference specific project details, tech choices, and achievements when relevant
- Be enthusiastic about his work but honest about limitations"""


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

DEMO_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Vineeth GPT</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#f7f6f2;
    --ink:#15151a;
    --muted:#8a8a8f;
    --line:#e3e1da;
    --navy:#1a1f5e;
    --magenta:#e8186d;
    --serif:'Fraunces', serif;
    --sans:'Inter', -apple-system, sans-serif;
  }
  html[data-theme="dark"]{
    --paper:#111114;
    --ink:#efeef0;
    --muted:#8d8d95;
    --line:#2a2a30;
    --navy:#8990e0;
    --magenta:#ff5c95;
  }
  *{box-sizing:border-box;}
  html,body{margin:0; padding:0;}
  body{
    font-family:var(--sans);
    color:var(--ink);
    background:var(--paper);
    min-height:100vh;
    position:relative;
    overflow-x:hidden;
    transition:background 0.25s ease, color 0.25s ease;
  }

  .bg{
    position:fixed;
    inset:0;
    z-index:0;
    pointer-events:none;
  }
  .bg::before, .bg::after{
    content:'';
    position:absolute;
    width:60vmax;
    height:60vmax;
    border-radius:50%;
    filter:blur(90px);
    opacity:0.10;
  }
  .bg::before{ background:var(--navy); top:-20vmax; left:-15vmax; }
  .bg::after{ background:var(--magenta); bottom:-25vmax; right:-18vmax; opacity:0.08; }
  .grain{
    position:fixed; inset:0; z-index:0; pointer-events:none;
    opacity:0.035; mix-blend-mode:multiply;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }

  .wrap{
    position:relative;
    z-index:1;
    max-width:620px;
    margin:0 auto;
    padding:72px 24px 60px;
  }

  header{ margin-bottom:44px; }
  .eyebrow{
    font-size:12px;
    letter-spacing:0.12em;
    text-transform:uppercase;
    color:var(--navy);
    font-weight:600;
    margin-bottom:14px;
  }
  .eyebrow-row{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
  }
  .theme-toggle{
    width:34px;
    height:34px;
    border-radius:50%;
    border:1px solid var(--line);
    background:transparent;
    color:var(--ink);
    display:flex;
    align-items:center;
    justify-content:center;
    cursor:pointer;
    flex-shrink:0;
    transition:border-color 0.2s ease, color 0.2s ease;
  }
  .theme-toggle:hover{ border-color:var(--magenta); color:var(--magenta); }
  .theme-toggle svg{ width:16px; height:16px; }
  .theme-toggle .icon-moon{ display:none; }
  html[data-theme="dark"] .theme-toggle .icon-sun{ display:none; }
  html[data-theme="dark"] .theme-toggle .icon-moon{ display:block; }

  h1{
    font-family:var(--serif);
    font-weight:500;
    font-size:clamp(32px, 6vw, 46px);
    line-height:1.08;
    letter-spacing:-0.01em;
    margin:0 0 12px;
  }
  h1 em{
    font-style:italic;
    color:var(--magenta);
  }
  p.sub{
    color:var(--muted);
    font-size:15.5px;
    line-height:1.6;
    max-width:440px;
    margin:0;
  }

  .composer{
    margin-top:38px;
    border-bottom:1.5px solid var(--ink);
    padding-bottom:10px;
    display:flex;
    align-items:flex-end;
    gap:14px;
    transition:border-color 0.2s ease;
  }
  .composer:focus-within{ border-color:var(--magenta); }
  textarea{
    flex:1;
    border:none;
    background:transparent;
    resize:none;
    font-family:var(--sans);
    font-size:17px;
    color:var(--ink);
    padding:6px 0;
    min-height:28px;
    max-height:120px;
  }
  textarea::placeholder{ color:var(--muted); }
  textarea:focus{ outline:none; }
  button.send{
    font-family:var(--sans);
    font-weight:600;
    font-size:13px;
    letter-spacing:0.02em;
    background:none;
    border:none;
    color:var(--ink);
    cursor:pointer;
    padding:6px 2px;
    white-space:nowrap;
    display:flex;
    align-items:center;
    gap:6px;
  }
  button.send:hover{ color:var(--magenta); }
  button.send:disabled{ color:var(--muted); cursor:default; }
  button.send svg{ width:14px; height:14px; }

  .suggestions{
    display:flex;
    flex-wrap:wrap;
    gap:8px 18px;
    margin-top:16px;
  }
  .sug{
    font-size:13px;
    color:var(--muted);
    cursor:pointer;
    border-bottom:1px solid transparent;
    transition:all 0.15s ease;
  }
  .sug:hover{ color:var(--navy); border-color:var(--navy); }

  .transcript{ margin-top:52px; }
  .entry{
    padding:28px 0;
    border-top:1px solid var(--line);
    animation:rise 0.4s ease both;
  }
  .entry:first-child{ border-top:1px solid var(--line); }
  @keyframes rise{
    from{ opacity:0; transform:translateY(6px); }
    to{ opacity:1; transform:translateY(0); }
  }
  .q-label{
    font-size:11px;
    font-weight:600;
    letter-spacing:0.1em;
    color:var(--magenta);
    margin-bottom:8px;
  }
  .q-text{
    font-family:var(--serif);
    font-size:19px;
    font-weight:500;
    line-height:1.35;
    margin-bottom:18px;
  }
  .a-label{
    font-size:11px;
    font-weight:600;
    letter-spacing:0.1em;
    color:var(--muted);
    margin-bottom:8px;
  }
  .a-text{
    font-size:16px;
    line-height:1.7;
    color:var(--ink);
    white-space:pre-wrap;
  }
  .a-text.error{ color:#b3261e; }
  .a-text.loading{ color:var(--muted); }

  .empty{
    color:var(--muted);
    font-size:14px;
    padding:36px 0;
    border-top:1px solid var(--line);
  }

  footer{
    margin-top:56px;
    padding-top:20px;
    border-top:1px solid var(--line);
    display:flex;
    flex-wrap:wrap;
    gap:18px;
    font-size:13px;
  }
  footer a{
    color:var(--muted);
    text-decoration:none;
    border-bottom:1px solid transparent;
  }
  footer a:hover{ color:var(--navy); border-color:var(--navy); }

  @media (max-width:480px){
    .wrap{ padding:52px 18px 40px; }
    .q-text{ font-size:17px; }
  }
</style>
</head>
<body>
  <div class="bg"></div>
  <div class="grain"></div>

  <div class="wrap">
    <header>
      <div class="eyebrow-row">
        <div class="eyebrow">Vineeth GPT</div>
        <button class="theme-toggle" id="themeToggle" aria-label="Toggle dark mode">
          <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
          <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1111.2 3 7 7 0 0021 12.8z"/></svg>
        </button>
      </div>
      <h1>Everything about<br><em>Vineeth</em>, answered.</h1>
      <p class="sub">Full-stack developer, UI/UX designer, and AI engineer. Ask about his projects, stack, or what he's looking for next.</p>

      <div class="composer">
        <textarea id="question" placeholder="What's his tech stack?" rows="1"></textarea>
        <button class="send" id="sendBtn">
          Ask
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
        </button>
      </div>

      <div class="suggestions">
        <span class="sug" data-q="What's his tech stack?">tech stack</span>
        <span class="sug" data-q="Tell me about ARIA.">ARIA</span>
        <span class="sug" data-q="What design work has he done?">design work</span>
        <span class="sug" data-q="What roles is he open to?">open roles</span>
      </div>
    </header>

    <div class="transcript" id="transcript">
      <div class="empty" id="emptyState">Nothing asked yet — start above, or pick a prompt.</div>
    </div>

    <footer>
      <a href="https://rickyfender.netlify.app" target="_blank">Portfolio</a>
      <a href="https://github.com/FenderRicky" target="_blank">GitHub</a>
      <a href="https://www.linkedin.com/in/vineeth-thadigotla-0569381b9/" target="_blank">LinkedIn</a>
      <a href="https://www.instagram.com/fender_ricky" target="_blank">Instagram</a>
    </footer>
  </div>

<script>
  const root = document.documentElement;
  const themeToggle = document.getElementById('themeToggle');

  function applyTheme(theme){
    root.setAttribute('data-theme', theme);
    localStorage.setItem('vineethgpt-theme', theme);
  }

  const savedTheme = localStorage.getItem('vineethgpt-theme');
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(savedTheme || (systemPrefersDark ? 'dark' : 'light'));

  themeToggle.addEventListener('click', () => {
    const current = root.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  const input = document.getElementById('question');
  const sendBtn = document.getElementById('sendBtn');
  const transcript = document.getElementById('transcript');
  const emptyState = document.getElementById('emptyState');

  function autoresize(){
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  }

  function addEntry(question){
    emptyState.style.display = 'none';
    const entry = document.createElement('div');
    entry.className = 'entry';
    entry.innerHTML = `
      <div class="q-label">QUESTION</div>
      <div class="q-text"></div>
      <div class="a-label">ANSWER</div>
      <div class="a-text loading">Thinking…</div>
    `;
    entry.querySelector('.q-text').textContent = question;
    transcript.prepend(entry);
    return entry.querySelector('.a-text');
  }

  async function ask(question){
    const q = (question || input.value).trim();
    if (!q) return;

    input.value = '';
    autoresize();
    sendBtn.disabled = true;

    const answerEl = addEntry(q);

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      
      answerEl.textContent = data.answer;
      answerEl.classList.remove('loading');
    } catch (err) {
      answerEl.textContent = 'Something went wrong: ' + err.message;
      answerEl.classList.remove('loading');
      answerEl.classList.add('error');
    } finally {
      sendBtn.disabled = false;
    }
  }

  sendBtn.addEventListener('click', () => ask());
  input.addEventListener('input', autoresize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  });
  document.querySelectorAll('.sug').forEach(el => {
    el.addEventListener('click', () => ask(el.dataset.q));
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def demo_page():
    return DEMO_HTML


@app.get("/api")
@app.get("/api/")
def root():
    return {"status": "ok", "message": "Ask Vineeth API is running"}


@app.post("/api/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be empty.")

    check_rate_limit()

    try:
        client = get_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.question},
        ]
        completion = None
        last_err = None
        for model in dict.fromkeys([GENERATION_MODEL, FALLBACK_MODEL]):  # primary, then fallback
            try:
                completion = client.chat.completions.create(model=model, messages=messages, temperature=0.6)
                break
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate_limit" in msg or "401" in msg or "authentication" in msg:
                    raise  # fallback model won't help; handled below
                last_err = e
        if completion is None:
            raise last_err

        # Safely extract the answer
        if not completion.choices or not completion.choices[0].message:
            raise HTTPException(
                status_code=500,
                detail="Groq returned an empty response. Try again."
            )

        answer = completion.choices[0].message.content
        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Groq returned empty content. Try again."
            )

        return QueryResponse(answer=answer)

    except HTTPException:
        raise
    except Exception as e:
        # Catch Groq errors, network errors, etc.
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Groq rate limit exceeded. Try again in a few minutes."
            )
        elif "401" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=500,
                detail="Groq API authentication failed. Check your API key."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"API error: {error_msg[:100]}"
            )


# ---------------------------------------------------------------------------
# IMPORTANT — the real usage cap
# ---------------------------------------------------------------------------
# The in-memory rate limiter above only protects a single warm serverless
# instance. Vercel can spin up multiple instances under load, each with its
# own counter, so it's a soft speed bump, not a hard guarantee.
#
# Groq's free tier is enforced server-side per API key/account (roughly
# 30 requests/min and ~1,000 requests/day for llama-3.3-70b-versatile as of
# 2026 — check console.groq.com for your account's current limits). That's
# the real backstop: once you hit it, Groq itself returns a 429, regardless
# of how many serverless instances Vercel spins up.

"""
Multi-Agent Research-to-Deck — Main Application
=================================================
True multi-agent pipeline with 6 independent agents.
Gamma integration for design + PPTX export.
"""

import os
import sys
import json
import uuid
import time
import httpx
import tempfile
import threading
import logging

from fastapi import FastAPI, Request, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

# Add parent directory to path so agents package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_pipeline
from agents.base import validate_required_keys

app = FastAPI(title="Multi-Agent Research-to-Deck")

# Restrictive CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # No cross-origin by default
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

GAMMA_API_KEY = os.environ.get("GAMMA_API_KEY", "")
APP_API_KEY = os.environ.get("APP_API_KEY", "")
TMPDIR = tempfile.mkdtemp()

# JOBS with timestamps for cleanup
JOBS: dict = {}  # job_id -> {"path": str, "created_at": float}
MAX_JOB_AGE = 3600  # 1 hour

# Simple in-memory rate limiting
RATE_LIMIT: dict = {}  # ip -> [timestamps]
RATE_LIMIT_MAX = 5  # requests per window
RATE_LIMIT_WINDOW = 300  # 5 minutes


# ============================================================
# Startup validation
# ============================================================

@app.on_event("startup")
async def startup():
    validate_required_keys()
    start_cleanup_loop()


# ============================================================
# Job cleanup — prevent unbounded growth
# ============================================================

def cleanup_old_jobs():
    now = time.time()
    expired = [jid for jid, info in JOBS.items() if now - info["created_at"] > MAX_JOB_AGE]
    for jid in expired:
        info = JOBS.pop(jid, None)
        if info and os.path.exists(info["path"]):
            try:
                os.remove(info["path"])
            except OSError:
                pass


def start_cleanup_loop():
    cleanup_old_jobs()
    t = threading.Timer(600, start_cleanup_loop)
    t.daemon = True
    t.start()


# ============================================================
# Authentication + Rate Limiting
# ============================================================

async def verify_api_key(x_api_key: str = Header(default="")):
    """Check API key if APP_API_KEY is set. Skip auth if not configured."""
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def check_rate_limit(request: Request):
    """Simple in-memory rate limiter by IP."""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    if ip in RATE_LIMIT:
        RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if t > window_start]
    else:
        RATE_LIMIT[ip] = []

    if len(RATE_LIMIT[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a few minutes.")

    RATE_LIMIT[ip].append(now)


# ============================================================
# Pydantic Models
# ============================================================

class DeckRequest(BaseModel):
    topic: str
    purpose: str = "inform"
    num_slides: int = Field(default=10, ge=5, le=25)
    audience: str = "general_business"
    theme: str = "professional"
    audience_role: Optional[str] = ""
    audience_familiarity: Optional[str] = "some"
    audience_motivation: Optional[str] = ""
    audience_objections: Optional[str] = ""
    desired_action: Optional[str] = ""
    narrative: Optional[str] = "pir"
    tone: Optional[str] = "authoritative"
    depth: Optional[str] = "standard"
    style: Optional[str] = ""
    constraints: Optional[str] = ""


# ============================================================
# Audience context builder
# ============================================================

AUDIENCE_PERSONAS = {
    "c_suite": "C-Suite executives who read slides beforehand, want P&L impact, 15-20 min attention span",
    "senior_management": "Senior managers who expect structured analysis with implementation feasibility",
    "external_peers": "Domain experts fact-checking in real time, want novel insights",
    "investors": "Investors focused on TAM, unit economics, unfair advantage and traction",
    "regulators": "Risk-averse regulators focused on compliance and independent verification",
    "general_business": "Mixed business audience, needs accessible but not patronizing content",
}


def build_audience_context(req: DeckRequest) -> str:
    persona = AUDIENCE_PERSONAS.get(req.audience, AUDIENCE_PERSONAS["general_business"])
    parts = [f"Persona: {persona}"]
    if req.audience_role:
        parts.append(f"Role: {req.audience_role}")
    if req.audience_familiarity:
        parts.append(f"Familiarity: {req.audience_familiarity}")
    if req.audience_motivation:
        parts.append(f"Motivation: {req.audience_motivation}")
    if req.audience_objections:
        parts.append(f"Likely objections: {req.audience_objections}")
    if req.desired_action:
        parts.append(f"Desired action: {req.desired_action}")
    if req.constraints:
        parts.append(f"Constraints: {req.constraints}")
    return " | ".join(parts)


# ============================================================
# Gamma Integration
# ============================================================

async def format_deck_for_gamma(deck_plan: dict) -> str:
    """Convert the orchestrator's JSON deck plan into structured text for Gamma."""
    parts = []
    for slide in deck_plan.get("slides", []):
        parts.append(f"---")
        parts.append(f"# {slide.get('title', 'Untitled')}")
        stype = slide.get("type", "context")

        if stype == "title":
            if slide.get("subtitle"):
                parts.append(f"*{slide['subtitle']}*")

        elif stype == "context":
            for card in slide.get("cards", []):
                parts.append(f"### {card.get('title', '')}")
                parts.append(card.get("body", ""))
            if slide.get("body"):
                parts.append(slide["body"])

        elif stype == "chart":
            if slide.get("chart_data"):
                parts.append("| Category | Value |")
                parts.append("|----------|-------|")
                for dp in slide["chart_data"]:
                    parts.append(f"| {dp.get('label', '')} | {dp.get('value', '')} |")
            if slide.get("body"):
                parts.append(slide["body"])

        elif stype == "deep_dive":
            if slide.get("body"):
                parts.append(slide["body"])
            for dp in slide.get("data_points", []):
                parts.append(f"**{dp.get('label', '')}:** {dp.get('value', '')}")

        elif stype == "comparison":
            if slide.get("left_column") and slide.get("right_column"):
                parts.append(f"**Left:** {slide['left_column']}")
                parts.append(f"**Right:** {slide['right_column']}")
            if slide.get("body"):
                parts.append(slide["body"])

        elif stype == "table":
            if slide.get("data_points"):
                parts.append("| Metric | Value |")
                parts.append("|--------|-------|")
                for dp in slide["data_points"]:
                    parts.append(f"| {dp.get('label', '')} | {dp.get('value', '')} |")

        elif stype == "recommendation":
            for action in slide.get("actions", []):
                parts.append(f"- **{action.get('action', '')}** — {action.get('timeline', '')} — {action.get('impact', '')}")

        elif stype == "takeaway":
            for bp in slide.get("bullet_points", []):
                parts.append(f"- {bp}")
            if slide.get("body"):
                parts.append(slide["body"])

        # Speaker notes as guidance
        if slide.get("speaker_notes"):
            parts.append(f"\n> Presenter: {slide['speaker_notes']}")

    return "\n\n".join(parts)


async def generate_via_gamma(deck_plan: dict, theme: str = "professional") -> dict:
    """Send deck plan to Gamma API for professional design + PPTX export."""
    if not GAMMA_API_KEY:
        return {"error": "No GAMMA_API_KEY set", "gamma_url": None, "file_path": None}

    input_text = await format_deck_for_gamma(deck_plan)
    tone_map = {
        "professional": "professional",
        "minimal": "casual",
        "bold": "bold",
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            "https://public-api.gamma.app/v1.0/generations",
            headers={
                "X-API-KEY": GAMMA_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "inputText": input_text,
                "format": "presentation",
                "textMode": "preserve",
                "exportAs": "pptx",
                "textOptions": {
                    "tone": tone_map.get(theme, "professional"),
                    "amount": "detailed",
                },
            },
        )
        resp.raise_for_status()
        gen = resp.json()
        gen_id = gen.get("id")

        if not gen_id:
            return {"error": "Gamma returned no generation ID", "gamma_url": None, "file_path": None}

        import asyncio
        for _ in range(40):
            await asyncio.sleep(3)
            status_resp = await client.get(
                f"https://public-api.gamma.app/v1.0/generations/{gen_id}",
                headers={"X-API-KEY": GAMMA_API_KEY},
            )
            status_resp.raise_for_status()
            status = status_resp.json()

            if status.get("status") == "completed":
                download_url = status.get("download_url") or status.get("exportUrl")
                gamma_url = status.get("url") or status.get("gammaUrl")

                file_path = None
                if download_url:
                    dl = await client.get(download_url)
                    dl.raise_for_status()
                    file_path = os.path.join(TMPDIR, f"{gen_id}.pptx")
                    with open(file_path, "wb") as f:
                        f.write(dl.content)

                return {
                    "gamma_url": gamma_url,
                    "file_path": file_path,
                    "gen_id": gen_id,
                }

            elif status.get("status") == "failed":
                return {"error": "Gamma generation failed", "gamma_url": None, "file_path": None}

        return {"error": "Gamma generation timed out", "gamma_url": None, "file_path": None}


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


@app.post("/api/generate")
async def generate(request: Request, _=Depends(verify_api_key)):
    """Stream the multi-agent pipeline progress, then generate via Gamma."""
    check_rate_limit(request)

    body = await request.json()
    req = DeckRequest(**body)
    audience_context = build_audience_context(req)

    async def event_stream():
        deck_plan = None

        async for event in run_pipeline(
            topic=req.topic,
            audience_context=audience_context,
            narrative=req.narrative or "pir",
            num_slides=req.num_slides,
            tone=req.tone or "authoritative",
            depth=req.depth or "standard",
        ):
            if event["event"] == "pipeline_complete":
                deck_plan = event["deck_plan"]
            yield f"data: {json.dumps(event)}\n\n"

        if deck_plan is None:
            yield f"data: {json.dumps({'event': 'error', 'message': 'Pipeline failed to produce a deck plan'})}\n\n"
            return

        # Format content for Gamma (used for both API and copy fallback)
        gamma_text = await format_deck_for_gamma(deck_plan)

        # Try Gamma API
        if GAMMA_API_KEY:
            yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'gamma', 'message': 'Gamma Design Engine: Creating professional slides...'})}\n\n"
            try:
                gamma_result = await generate_via_gamma(deck_plan, req.theme)
                if gamma_result.get("error"):
                    yield f"data: {json.dumps({'event': 'agent_error', 'agent': 'gamma', 'message': gamma_result['error']})}\n\n"
                    yield f"data: {json.dumps({'event': 'complete', 'gamma_content': gamma_text})}\n\n"
                else:
                    job_id = str(uuid.uuid4())
                    if gamma_result.get("file_path"):
                        JOBS[job_id] = {"path": gamma_result["file_path"], "created_at": time.time()}
                    yield f"data: {json.dumps({'event': 'agent_complete', 'agent': 'gamma', 'message': 'Gamma design complete'})}\n\n"
                    yield f"data: {json.dumps({'event': 'complete', 'job_id': job_id, 'gamma_url': gamma_result.get('gamma_url')})}\n\n"
            except Exception as e:
                logger.exception("Gamma API error")
                yield f"data: {json.dumps({'event': 'agent_error', 'agent': 'gamma', 'message': 'Gamma design failed'})}\n\n"
                yield f"data: {json.dumps({'event': 'complete', 'gamma_content': gamma_text})}\n\n"
        else:
            yield f"data: {json.dumps({'event': 'complete', 'gamma_content': gamma_text})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/download/{job_id}")
async def download(job_id: str, _=Depends(verify_api_key)):
    info = JOBS.get(job_id)
    if not info or not os.path.exists(info["path"]):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(info["path"], filename="presentation.pptx", media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")


@app.get("/health")
async def health():
    return {"status": "ok", "agents": 6, "gamma": bool(GAMMA_API_KEY)}


# ============================================================
# Frontend UI
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Agent Research-to-Deck</title>
<style>
  :root {
    --bg: #0a0b10; --surface: #12131a; --surface2: #1a1b25;
    --border: #2a2b3a; --text: #e0e0e8; --text-dim: #a0a0b8;
    --accent: #7c6aff; --accent-glow: rgba(124,106,255,0.15);
    --green: #34d399; --yellow: #fbbf24; --red: #f87171; --blue: #60a5fa;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Inter',-apple-system,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
  .skip-link { position:absolute; top:-40px; left:0; background:var(--accent); color:#fff; padding:8px; z-index:100; }
  .skip-link:focus { top:0; }
  .container { max-width:720px; margin:0 auto; padding:2rem 1.5rem; }
  h1 { font-size:1.6rem; font-weight:700; text-align:center;
    background:linear-gradient(135deg,#7c6aff,#60a5fa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .subtitle { text-align:center; color:var(--text-dim); font-size:0.85rem; margin:0.3rem 0 0.8rem; }
  .badge-row { display:flex; gap:0.4rem; justify-content:center; flex-wrap:wrap; margin-bottom:2rem; }
  .badge { display:inline-flex; align-items:center; gap:0.25rem; padding:0.2rem 0.55rem;
    border-radius:999px; font-size:0.7rem; font-weight:500;
    background:var(--surface2); border:1px solid var(--border); color:var(--text-dim); }
  .dot { width:5px; height:5px; border-radius:50%; }

  /* Form */
  .form-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; }
  label { font-size:0.8rem; color:var(--text-dim); display:block; margin-bottom:0.3rem; }
  input, select, textarea { width:100%; background:var(--bg); color:var(--text); border:1px solid var(--border);
    border-radius:8px; padding:0.7rem; font-size:0.9rem; margin-bottom:0.8rem; }
  input:focus, select:focus, textarea:focus { outline:none; border-color:var(--accent); }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; }
  .row3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.75rem; }
  .collapse-btn { background:none; border:none; color:var(--accent); font-size:0.8rem; cursor:pointer;
    padding:0.3rem 0; margin-bottom:0.5rem; text-align:left; min-height:44px; }
  .collapse-content { display:none; }
  .collapse-content.open { display:block; }
  .topic-error { color:var(--red); font-size:0.8rem; display:none; margin-bottom:0.5rem; }

  .btn { background:var(--accent); color:#fff; border:none; padding:0.75rem 2rem; border-radius:8px;
    font-size:0.95rem; font-weight:600; cursor:pointer; width:100%; margin-top:0.5rem; }
  .btn:hover { opacity:0.85; }
  .btn:disabled { opacity:0.4; cursor:not-allowed; }

  /* Agent pipeline status */
  .pipeline { display:none; background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:1.25rem; margin-bottom:1.5rem; }
  .pipeline.active { display:block; }
  .pipeline h2 { font-size:0.9rem; color:var(--text-dim); margin-bottom:0.75rem; }
  .agent-row { display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0;
    border-bottom:1px solid var(--border); font-size:0.82rem; }
  .agent-row:last-child { border-bottom:none; }
  .agent-dot { width:8px; height:8px; border-radius:50%; background:var(--border); flex-shrink:0; }
  .agent-dot.running { background:var(--accent); animation:pulse 1s infinite; }
  .agent-dot.done { background:var(--green); }
  .agent-dot.error { background:var(--red); }
  .agent-dot.rejected { background:var(--yellow); }
  .agent-dot.validated { background:var(--green); }
  .agent-label { font-weight:600; min-width:90px; }
  .agent-msg { color:var(--text-dim); flex:1; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

  /* Result */
  .result { display:none; background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:1.5rem; text-align:center; }
  .result.active { display:block; }
  .dl-btn, .copy-btn { display:inline-block; background:linear-gradient(135deg,var(--accent),var(--green));
    color:#fff; padding:0.75rem 2rem; border-radius:8px; font-weight:600; text-decoration:none;
    margin-top:1rem; cursor:pointer; border:none; font-size:0.95rem; }
  .copy-btn:hover { opacity:0.85; }
  .gamma-content { width:100%; min-height:300px; max-height:500px; background:var(--bg); color:var(--text);
    border:1px solid var(--border); border-radius:8px; padding:1rem; font-family:monospace;
    font-size:0.8rem; resize:vertical; margin-top:1rem; white-space:pre-wrap; }
  .result-header { color:var(--green); margin-bottom:0.5rem; }
  .result-hint { color:var(--text-dim); font-size:0.8rem; margin-top:0.5rem; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
<a href="#formCard" class="skip-link">Skip to form</a>
<header class="container" style="padding-bottom:0;">
  <h1>Multi-Agent Research-to-Deck</h1>
  <p class="subtitle">6 independent AI agents build your presentation in parallel</p>
  <div class="badge-row" aria-hidden="true">
    <span class="badge"><span class="dot" style="background:var(--blue)"></span>Researcher</span>
    <span class="badge"><span class="dot" style="background:var(--green)"></span>Brief</span>
    <span class="badge"><span class="dot" style="background:var(--yellow)"></span>Validator</span>
    <span class="badge"><span class="dot" style="background:var(--accent)"></span>Architect</span>
    <span class="badge"><span class="dot" style="background:var(--red)"></span>Writer</span>
    <span class="badge"><span class="dot" style="background:#c084fc"></span>Reviewer</span>
  </div>
</header>

<main class="container" style="padding-top:0;">
  <form class="form-card" id="formCard" onsubmit="return false;">
    <label for="topic">Research Topic</label>
    <input type="text" id="topic" required aria-required="true" placeholder="e.g. The future of AI agents in enterprise software">
    <div class="topic-error" id="topicError" role="alert">Please enter a research topic</div>

    <div class="row">
      <div><label for="purpose">Purpose</label>
        <select id="purpose">
          <option value="inform">Inform</option><option value="persuade">Persuade</option>
          <option value="sell">Sell</option><option value="educate">Educate</option>
          <option value="report">Report</option><option value="pitch">Pitch</option>
        </select></div>
      <div><label for="numSlides">Slides</label>
        <input type="number" id="numSlides" value="10" min="5" max="25"></div>
    </div>

    <div class="row">
      <div><label for="audience">Audience</label>
        <select id="audience">
          <option value="general_business">General Business</option>
          <option value="c_suite">C-Suite</option>
          <option value="senior_management">Senior Management</option>
          <option value="external_peers">External Peers</option>
          <option value="investors">Investors</option>
          <option value="regulators">Regulators</option>
        </select></div>
      <div><label for="theme">Design Theme</label>
        <select id="theme">
          <option value="professional">Professional</option>
          <option value="minimal">Minimal</option>
          <option value="bold">Bold</option>
        </select></div>
    </div>

    <button type="button" class="collapse-btn" aria-expanded="false" aria-controls="audienceDetail" data-label="Audience Detail" onclick="toggleSection(this,'audienceDetail')">+ Audience Detail</button>
    <div class="collapse-content" id="audienceDetail">
      <div class="row">
        <div><label for="audienceRole">Role</label><input type="text" id="audienceRole" placeholder="e.g. VP Engineering"></div>
        <div><label for="audienceFamiliarity">Familiarity</label>
          <select id="audienceFamiliarity">
            <option value="none">None</option><option value="some" selected>Some</option>
            <option value="expert">Expert</option>
          </select></div>
      </div>
      <label for="audienceMotivation">Motivation</label><input type="text" id="audienceMotivation" placeholder="What brought them to this meeting?">
      <label for="audienceObjections">Likely Objections</label><input type="text" id="audienceObjections" placeholder="What will they push back on?">
      <label for="desiredAction">Desired Action</label><input type="text" id="desiredAction" placeholder="What should they do after?">
    </div>

    <button type="button" class="collapse-btn" aria-expanded="false" aria-controls="narrativeDetail" data-label="Narrative & Tone" onclick="toggleSection(this,'narrativeDetail')">+ Narrative & Tone</button>
    <div class="collapse-content" id="narrativeDetail">
      <div class="row3">
        <div><label for="narrative">Narrative Arc</label>
          <select id="narrative">
            <option value="pir">PIR (McKinsey)</option>
            <option value="scqa">SCQA (BCG)</option>
            <option value="change">Change Story</option>
          </select></div>
        <div><label for="tone">Tone</label>
          <select id="tone">
            <option value="authoritative">Authoritative</option>
            <option value="collaborative">Collaborative</option>
            <option value="provocative">Provocative</option>
            <option value="neutral">Neutral</option>
            <option value="inspirational">Inspirational</option>
          </select></div>
        <div><label for="depth">Depth</label>
          <select id="depth">
            <option value="overview">Overview</option>
            <option value="standard" selected>Standard</option>
            <option value="deep">Deep Dive</option>
          </select></div>
      </div>
      <label for="constraints">Constraints</label>
      <textarea id="constraints" rows="2" placeholder="Any special requirements..."></textarea>
    </div>

    <button type="button" class="btn" id="genBtn" onclick="generate()">Generate Deck</button>
  </form>

  <div class="pipeline" id="pipeline" role="region" aria-label="Pipeline progress">
    <h2>Agent Pipeline</h2>
    <div id="agentLog" role="log" aria-live="polite" aria-label="Agent status updates"></div>
  </div>

  <div class="result" id="result" role="region" aria-label="Result"></div>
</main>

<script>
function toggleSection(btn, id) {
  var el = document.getElementById(id);
  el.classList.toggle('open');
  var isOpen = el.classList.contains('open');
  btn.setAttribute('aria-expanded', isOpen);
  btn.textContent = (isOpen ? '- ' : '+ ') + btn.dataset.label;
}

function addAgentRow(agent, msg, status) {
  var log = document.getElementById('agentLog');
  var id = 'row-' + agent + '-' + Date.now();
  var row = document.createElement('div');
  row.className = 'agent-row';
  row.id = id;
  row.innerHTML = '<div class="agent-dot '+status+'" aria-hidden="true"></div><span class="agent-label">'+esc(agent)+'</span><span class="agent-msg">'+esc(msg)+'</span>';
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return id;
}

function updateLastRow(status, msg) {
  var rows = document.querySelectorAll('.agent-row');
  if (!rows.length) return;
  var last = rows[rows.length - 1];
  last.querySelector('.agent-dot').className = 'agent-dot ' + status;
  if (msg) last.querySelector('.agent-msg').textContent = msg;
}

var controller;

async function generate() {
  var topicEl = document.getElementById('topic');
  var topic = topicEl.value.trim();
  var errEl = document.getElementById('topicError');

  if (!topic) {
    topicEl.setAttribute('aria-invalid', 'true');
    errEl.style.display = 'block';
    topicEl.focus();
    return;
  }
  errEl.style.display = 'none';
  topicEl.removeAttribute('aria-invalid');

  var btn = document.getElementById('genBtn');
  var pipeline = document.getElementById('pipeline');
  var result = document.getElementById('result');

  controller = new AbortController();
  btn.disabled = true;
  btn.textContent = 'Agents running...';
  pipeline.classList.add('active');
  result.classList.remove('active'); result.innerHTML = '';
  document.getElementById('agentLog').innerHTML = '';

  var body = {
    topic: topic,
    purpose: document.getElementById('purpose').value,
    num_slides: parseInt(document.getElementById('numSlides').value),
    audience: document.getElementById('audience').value,
    theme: document.getElementById('theme').value,
    audience_role: document.getElementById('audienceRole') ? document.getElementById('audienceRole').value : '',
    audience_familiarity: document.getElementById('audienceFamiliarity') ? document.getElementById('audienceFamiliarity').value : 'some',
    audience_motivation: document.getElementById('audienceMotivation') ? document.getElementById('audienceMotivation').value : '',
    audience_objections: document.getElementById('audienceObjections') ? document.getElementById('audienceObjections').value : '',
    desired_action: document.getElementById('desiredAction') ? document.getElementById('desiredAction').value : '',
    narrative: document.getElementById('narrative') ? document.getElementById('narrative').value : 'pir',
    tone: document.getElementById('tone') ? document.getElementById('tone').value : 'authoritative',
    depth: document.getElementById('depth') ? document.getElementById('depth').value : 'standard',
    constraints: document.getElementById('constraints') ? document.getElementById('constraints').value : ''
  };

  try {
    var res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
      signal: controller.signal
    });

    if (!res.ok) {
      var errData = await res.json().catch(function() { return {}; });
      throw new Error(errData.detail || 'Server error ' + res.status);
    }

    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';

    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, {stream:true});
      var lines = buffer.split('\\n');
      buffer = lines.pop();
      for (var i = 0; i < lines.length; i++) {
        if (!lines[i].startsWith('data: ')) continue;
        try { handleEvent(JSON.parse(lines[i].slice(6))); } catch(e) {}
      }
    }
  } catch(e) {
    if (e.name !== 'AbortError') {
      result.classList.add('active');
      result.innerHTML = '<p style="color:var(--red)">'+esc(e.message || 'Network error')+'</p>';
    }
  }
  btn.disabled = false; btn.textContent = 'Generate Deck';
  controller = null;
}

function handleEvent(e) {
  switch(e.event) {
    case 'agent_start':
      addAgentRow(e.agent || 'system', e.message, 'running');
      break;
    case 'agent_complete':
      updateLastRow('done', e.message);
      break;
    case 'agent_error':
      updateLastRow('error', e.message);
      break;
    case 'validating':
      addAgentRow('validator', e.message, 'running');
      break;
    case 'validated':
      updateLastRow('validated', e.message);
      break;
    case 'rejected':
      updateLastRow('rejected', e.message);
      break;
    case 'heartbeat':
      updateLastRow('running', e.message);
      break;
    case 'warning':
      addAgentRow('warning', e.message, 'rejected');
      break;
    case 'pipeline_complete':
      addAgentRow('pipeline', e.message, 'done');
      break;
    case 'complete': {
      var r = document.getElementById('result');
      r.classList.add('active');
      var html = '';
      if (e.gamma_url) {
        html += '<h2 class="result-header">Deck Ready</h2>';
        if (e.job_id) html += '<a class="dl-btn" href="/api/download/'+esc(e.job_id)+'" style="margin-right:0.75rem;">Download PPTX</a>';
        html += '<a class="dl-btn" href="'+esc(e.gamma_url)+'" target="_blank">Edit in Gamma</a>';
      } else if (e.gamma_content) {
        html += '<h2 class="result-header">Deck Ready</h2>';
        html += '<p class="result-hint">Copy and paste into Gamma to generate your slides.</p>';
        html += '<textarea class="gamma-content" id="gammaContent" readonly aria-label="Deck content for Gamma">'+esc(e.gamma_content)+'</textarea>';
        html += '<button class="copy-btn" onclick="copyGamma()">Copy to Clipboard</button>';
        html += '<span id="copyMsg" style="margin-left:0.75rem;color:var(--green);font-size:0.85rem;display:none;" role="status">Copied!</span>';
      } else {
        html += '<h2 class="result-header">Deck Ready</h2>';
        if (e.job_id) html += '<a class="dl-btn" href="/api/download/'+esc(e.job_id)+'">Download PPTX</a>';
      }
      r.innerHTML = html;
      break;
    }
    case 'error':
      addAgentRow('error', e.message, 'error');
      break;
  }
}

function esc(s) { var d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

function copyGamma() {
  var ta = document.getElementById('gammaContent');
  if (!ta) return;
  ta.select();
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(ta.value).then(function() {
      var msg = document.getElementById('copyMsg');
      if (msg) { msg.style.display = 'inline'; setTimeout(function() { msg.style.display = 'none'; }, 2000); }
    }).catch(function() {
      document.execCommand('copy');
    });
  } else {
    document.execCommand('copy');
  }
}

window.addEventListener('beforeunload', function(e) {
  if (controller) { e.preventDefault(); e.returnValue = ''; }
});
</script>
</body>
</html>
"""

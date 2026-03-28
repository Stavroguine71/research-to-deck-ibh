"""
Multi-Agent Research-to-Deck — Main Application
=================================================
True multi-agent pipeline with 6 independent agents:
  1. Researcher — 4 parallel Tavily queries
  2. Brief Writer — synthesizes research
  3. Validator — quality gates between passes (NEW)
  4. Architect — designs slide skeleton
  5. Writer — fills full content
  6. Reviewer — partner-level quality gate with rewrites

Gamma integration preserved for design + PPTX export.
"""

import os
import sys
import json
import uuid
import httpx
import tempfile

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# Add parent directory to path so agents package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_pipeline

app = FastAPI(title="Multi-Agent Research-to-Deck")

GAMMA_API_KEY = os.environ.get("GAMMA_API_KEY", "")
TMPDIR = tempfile.mkdtemp()
JOBS: dict = {}


# ============================================================
# Pydantic Models (same as your existing version)
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
# Audience context builder (same as your existing version)
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
# Gamma Integration (preserved from your existing version)
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

    content = await format_deck_for_gamma(deck_plan)
    tone_map = {
        "professional": "professional and polished",
        "minimal": "clean and minimal",
        "bold": "bold and high-contrast",
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        # Start generation
        resp = await client.post(
            "https://api.gamma.app/api/v1/generations",
            headers={
                "Authorization": f"Bearer {GAMMA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "content": content,
                "tone": tone_map.get(theme, "professional and polished"),
                "output_format": "presentation",
            },
        )
        resp.raise_for_status()
        gen = resp.json()
        gen_id = gen.get("id")

        if not gen_id:
            return {"error": "Gamma returned no generation ID", "gamma_url": None, "file_path": None}

        # Poll for completion
        import asyncio
        for _ in range(40):  # 40 * 3s = 120s max
            await asyncio.sleep(3)
            status_resp = await client.get(
                f"https://api.gamma.app/api/v1/generations/{gen_id}",
                headers={"Authorization": f"Bearer {GAMMA_API_KEY}"},
            )
            status_resp.raise_for_status()
            status = status_resp.json()

            if status.get("status") == "completed":
                download_url = status.get("download_url")
                gamma_url = status.get("url")

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
async def generate(request: Request):
    """Stream the multi-agent pipeline progress, then generate via Gamma."""
    body = await request.json()
    req = DeckRequest(**body)
    audience_context = build_audience_context(req)

    async def event_stream():
        deck_plan = None

        # Run the multi-agent pipeline with streaming events
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

        # Gamma design phase
        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'gamma', 'message': 'Gamma Design Engine: Creating professional slides...'})}\n\n"

        try:
            gamma_result = await generate_via_gamma(deck_plan, req.theme)

            if gamma_result.get("error"):
                yield f"data: {json.dumps({'event': 'agent_error', 'agent': 'gamma', 'message': gamma_result['error']})}\n\n"

                # Fallback to python-pptx
                yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'pptx_fallback', 'message': 'Falling back to python-pptx generation...'})}\n\n"
                try:
                    from deck_builder import build_deck
                    job_id = str(uuid.uuid4())
                    output_path = os.path.join(TMPDIR, f"{job_id}.pptx")
                    build_deck(deck_plan, output_path, req.theme)
                    JOBS[job_id] = output_path
                    yield f"data: {json.dumps({'event': 'complete', 'job_id': job_id, 'gamma_url': None})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'event': 'error', 'message': f'Fallback also failed: {str(e)}'})}\n\n"
            else:
                job_id = str(uuid.uuid4())
                if gamma_result.get("file_path"):
                    JOBS[job_id] = gamma_result["file_path"]
                yield f"data: {json.dumps({'event': 'agent_complete', 'agent': 'gamma', 'message': 'Gamma design complete'})}\n\n"
                yield f"data: {json.dumps({'event': 'complete', 'job_id': job_id, 'gamma_url': gamma_result.get('gamma_url')})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': f'Gamma error: {str(e)}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    path = JOBS.get(job_id)
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, filename="presentation.pptx", media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")


@app.get("/health")
async def health():
    return {"status": "ok", "agents": 6, "gamma": bool(GAMMA_API_KEY)}


# ============================================================
# Frontend UI with real-time agent status tracking
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
    --border: #2a2b3a; --text: #e0e0e8; --text-dim: #8888a0;
    --accent: #7c6aff; --accent-glow: rgba(124,106,255,0.15);
    --green: #34d399; --yellow: #fbbf24; --red: #f87171; --blue: #60a5fa;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Inter',-apple-system,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
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
    padding:0.3rem 0; margin-bottom:0.5rem; text-align:left; }
  .collapse-content { display:none; }
  .collapse-content.open { display:block; }

  .btn { background:var(--accent); color:#fff; border:none; padding:0.75rem 2rem; border-radius:8px;
    font-size:0.95rem; font-weight:600; cursor:pointer; width:100%; margin-top:0.5rem; }
  .btn:hover { opacity:0.85; }
  .btn:disabled { opacity:0.4; cursor:not-allowed; }

  /* Agent pipeline status */
  .pipeline { display:none; background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:1.25rem; margin-bottom:1.5rem; }
  .pipeline.active { display:block; }
  .pipeline h3 { font-size:0.9rem; color:var(--text-dim); margin-bottom:0.75rem; }
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
  .dl-btn { display:inline-block; background:linear-gradient(135deg,var(--accent),var(--green));
    color:#fff; padding:0.75rem 2rem; border-radius:8px; font-weight:600; text-decoration:none;
    margin-top:1rem; }
  .gamma-link { display:block; margin-top:0.75rem; color:var(--accent); font-size:0.85rem; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
<div class="container">
  <h1>Multi-Agent Research-to-Deck</h1>
  <p class="subtitle">6 independent AI agents build your presentation in parallel</p>
  <div class="badge-row">
    <span class="badge"><span class="dot" style="background:var(--blue)"></span>Researcher</span>
    <span class="badge"><span class="dot" style="background:var(--green)"></span>Brief</span>
    <span class="badge"><span class="dot" style="background:var(--yellow)"></span>Validator</span>
    <span class="badge"><span class="dot" style="background:var(--accent)"></span>Architect</span>
    <span class="badge"><span class="dot" style="background:var(--red)"></span>Writer</span>
    <span class="badge"><span class="dot" style="background:#c084fc"></span>Reviewer</span>
  </div>

  <div class="form-card" id="formCard">
    <label>Research Topic</label>
    <input type="text" id="topic" placeholder="e.g. The future of AI agents in enterprise software">

    <div class="row">
      <div><label>Purpose</label>
        <select id="purpose">
          <option value="inform">Inform</option><option value="persuade">Persuade</option>
          <option value="sell">Sell</option><option value="educate">Educate</option>
          <option value="report">Report</option><option value="pitch">Pitch</option>
        </select></div>
      <div><label>Slides</label>
        <input type="number" id="numSlides" value="10" min="5" max="25"></div>
    </div>

    <div class="row">
      <div><label>Audience</label>
        <select id="audience">
          <option value="general_business">General Business</option>
          <option value="c_suite">C-Suite</option>
          <option value="senior_management">Senior Management</option>
          <option value="external_peers">External Peers</option>
          <option value="investors">Investors</option>
          <option value="regulators">Regulators</option>
        </select></div>
      <div><label>Design Theme</label>
        <select id="theme">
          <option value="professional">Professional</option>
          <option value="minimal">Minimal</option>
          <option value="bold">Bold</option>
        </select></div>
    </div>

    <button class="collapse-btn" onclick="toggle('audienceDetail')">+ Audience Detail</button>
    <div class="collapse-content" id="audienceDetail">
      <div class="row">
        <div><label>Role</label><input type="text" id="audienceRole" placeholder="e.g. VP Engineering"></div>
        <div><label>Familiarity</label>
          <select id="audienceFamiliarity">
            <option value="none">None</option><option value="some" selected>Some</option>
            <option value="expert">Expert</option>
          </select></div>
      </div>
      <label>Motivation</label><input type="text" id="audienceMotivation" placeholder="What brought them to this meeting?">
      <label>Likely Objections</label><input type="text" id="audienceObjections" placeholder="What will they push back on?">
      <label>Desired Action</label><input type="text" id="desiredAction" placeholder="What should they do after?">
    </div>

    <button class="collapse-btn" onclick="toggle('narrativeDetail')">+ Narrative & Tone</button>
    <div class="collapse-content" id="narrativeDetail">
      <div class="row3">
        <div><label>Narrative Arc</label>
          <select id="narrative">
            <option value="pir">PIR (McKinsey)</option>
            <option value="scqa">SCQA (BCG)</option>
            <option value="change">Change Story</option>
          </select></div>
        <div><label>Tone</label>
          <select id="tone">
            <option value="authoritative">Authoritative</option>
            <option value="collaborative">Collaborative</option>
            <option value="provocative">Provocative</option>
            <option value="neutral">Neutral</option>
            <option value="inspirational">Inspirational</option>
          </select></div>
        <div><label>Depth</label>
          <select id="depth">
            <option value="overview">Overview</option>
            <option value="standard" selected>Standard</option>
            <option value="deep">Deep Dive</option>
          </select></div>
      </div>
      <label>Constraints</label>
      <textarea id="constraints" rows="2" placeholder="Any special requirements..."></textarea>
    </div>

    <button class="btn" id="genBtn" onclick="generate()">Generate Deck</button>
  </div>

  <div class="pipeline" id="pipeline">
    <h3>Agent Pipeline</h3>
    <div id="agentLog"></div>
  </div>

  <div class="result" id="result"></div>
</div>

<script>
function toggle(id) { document.getElementById(id).classList.toggle('open'); }

function addAgentRow(agent, msg, status) {
  const log = document.getElementById('agentLog');
  const id = 'row-' + agent + '-' + Date.now();
  const row = document.createElement('div');
  row.className = 'agent-row';
  row.id = id;
  row.innerHTML = '<div class="agent-dot '+status+'"></div><span class="agent-label">'+esc(agent)+'</span><span class="agent-msg">'+esc(msg)+'</span>';
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return id;
}

function updateLastRow(status, msg) {
  const rows = document.querySelectorAll('.agent-row');
  if (!rows.length) return;
  const last = rows[rows.length - 1];
  last.querySelector('.agent-dot').className = 'agent-dot ' + status;
  if (msg) last.querySelector('.agent-msg').textContent = msg;
}

async function generate() {
  const topic = document.getElementById('topic').value.trim();
  if (!topic) return;

  const btn = document.getElementById('genBtn');
  const pipeline = document.getElementById('pipeline');
  const result = document.getElementById('result');

  btn.disabled = true; btn.textContent = 'Agents running...';
  pipeline.classList.add('active');
  result.classList.remove('active'); result.innerHTML = '';
  document.getElementById('agentLog').innerHTML = '';

  const body = {
    topic,
    purpose: document.getElementById('purpose').value,
    num_slides: parseInt(document.getElementById('numSlides').value),
    audience: document.getElementById('audience').value,
    theme: document.getElementById('theme').value,
    audience_role: document.getElementById('audienceRole')?.value || '',
    audience_familiarity: document.getElementById('audienceFamiliarity')?.value || 'some',
    audience_motivation: document.getElementById('audienceMotivation')?.value || '',
    audience_objections: document.getElementById('audienceObjections')?.value || '',
    desired_action: document.getElementById('desiredAction')?.value || '',
    narrative: document.getElementById('narrative')?.value || 'pir',
    tone: document.getElementById('tone')?.value || 'authoritative',
    depth: document.getElementById('depth')?.value || 'standard',
    constraints: document.getElementById('constraints')?.value || '',
  };

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try { handleEvent(JSON.parse(line.slice(6))); } catch(e) {}
      }
    }
  } catch(e) {
    result.classList.add('active');
    result.innerHTML = '<p style="color:var(--red)">'+esc(e.message)+'</p>';
  }
  btn.disabled = false; btn.textContent = 'Generate Deck';
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
      const r = document.getElementById('result');
      r.classList.add('active');
      let html = '<h2 style="color:var(--green)">Deck Ready</h2>';
      if (e.job_id) html += '<a class="dl-btn" href="/api/download/'+e.job_id+'">Download PPTX</a>';
      if (e.gamma_url) html += '<a class="gamma-link" href="'+e.gamma_url+'" target="_blank">Edit in Gamma →</a>';
      r.innerHTML = html;
      break;
    }
    case 'error':
      addAgentRow('error', e.message, 'error');
      break;
  }
}

function esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
</script>
</body>
</html>
"""

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
import asyncio
import httpx
import tempfile
import logging

from fastapi import FastAPI, Request, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal

logger = logging.getLogger(__name__)

try:
    from agents.orchestrator import run_pipeline
    from agents.base import validate_required_keys
except ImportError:
    # Fallback: add parent directory if running outside installed package
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


# Security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # CSP: allow inline styles/scripts (needed for single-page HTML), fonts from Google
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'"
    )
    return response


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
    if not APP_API_KEY:
        raise RuntimeError("APP_API_KEY must be set to secure the API. Set this env var before starting.")
    asyncio.create_task(cleanup_loop())


# ============================================================
# Job cleanup — async loop (no threading race condition)
# ============================================================

async def cleanup_loop():
    """Runs on the same event loop as the app — no race conditions."""
    while True:
        await asyncio.sleep(600)
        now = time.time()
        expired = [jid for jid, info in JOBS.items() if now - info["created_at"] > MAX_JOB_AGE]
        for jid in expired:
            info = JOBS.pop(jid, None)
            if info and os.path.exists(info["path"]):
                try:
                    os.remove(info["path"])
                except OSError:
                    pass
        # Clean stale rate limit entries
        window_start = now - RATE_LIMIT_WINDOW
        stale_ips = [ip for ip, ts in RATE_LIMIT.items() if not any(t > window_start for t in ts)]
        for ip in stale_ips:
            RATE_LIMIT.pop(ip, None)


# ============================================================
# Authentication + Rate Limiting
# ============================================================

async def verify_api_key(x_api_key: str = Header(default="")):
    """Check API key if APP_API_KEY is set. Skip auth if not configured."""
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


MAX_RATE_ENTRIES = 10000


def get_client_ip(request: Request) -> str:
    """Use direct TCP connection IP — never trust X-Forwarded-For to prevent spoofing."""
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request):
    """In-memory rate limiter by IP. Returns remaining count for headers."""
    ip = get_client_ip(request)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    if ip in RATE_LIMIT:
        RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if t > window_start]
    else:
        RATE_LIMIT[ip] = []

    if len(RATE_LIMIT[ip]) >= RATE_LIMIT_MAX:
        remaining = max(1, int(RATE_LIMIT_WINDOW - (now - RATE_LIMIT[ip][0])))
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {remaining} seconds.")

    RATE_LIMIT[ip].append(now)

    # Evict oldest entries if dict grows too large (DoS protection)
    if len(RATE_LIMIT) > MAX_RATE_ENTRIES:
        oldest_ip = min(RATE_LIMIT, key=lambda k: RATE_LIMIT[k][0] if RATE_LIMIT[k] else 0)
        del RATE_LIMIT[oldest_ip]


# ============================================================
# Pydantic Models
# ============================================================

class DeckRequest(BaseModel):
    topic: str = Field(..., max_length=500)
    purpose: Literal["inform", "persuade", "sell", "educate", "report", "pitch"] = "inform"
    num_slides: int = Field(default=10, ge=5, le=25)
    audience: Literal["general_business", "c_suite", "senior_management", "external_peers", "investors", "regulators"] = "general_business"
    theme: Literal["professional", "minimal", "bold"] = "professional"
    audience_role: Optional[str] = Field(default="", max_length=200)
    audience_familiarity: Optional[Literal["none", "some", "expert"]] = "some"
    audience_motivation: Optional[str] = Field(default="", max_length=300)
    audience_objections: Optional[str] = Field(default="", max_length=300)
    desired_action: Optional[str] = Field(default="", max_length=300)
    narrative: Optional[Literal["pir", "scqa", "change"]] = "pir"
    tone: Optional[Literal["authoritative", "collaborative", "provocative", "neutral", "inspirational"]] = "authoritative"
    depth: Optional[Literal["overview", "standard", "deep"]] = "standard"
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
    gamma_key = os.environ.get("GAMMA_API_KEY", "")
    if not gamma_key:
        return {"error": "GAMMA_API_KEY not configured", "gamma_url": None, "file_path": None}

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
                "X-API-KEY": gamma_key,
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

        for _ in range(40):
            await asyncio.sleep(3)
            try:
                status_resp = await client.get(
                    f"https://public-api.gamma.app/v1.0/generations/{gen_id}",
                    headers={"X-API-KEY": gamma_key},
                )
                status_resp.raise_for_status()
                status = status_resp.json()
            except (httpx.HTTPError, httpx.TransportError) as e:
                logger.warning(f"Gamma poll transient error: {e}")
                continue

            if status.get("status") == "completed":
                download_url = status.get("download_url") or status.get("exportUrl")
                gamma_url = status.get("url") or status.get("gammaUrl")

                file_path = None
                if download_url:
                    from urllib.parse import urlparse
                    parsed = urlparse(download_url)
                    ALLOWED_HOSTS = {"gamma.app", "public-api.gamma.app", "cdn.gamma.app"}
                    if parsed.hostname not in ALLOWED_HOSTS:
                        return {"error": "Untrusted download URL from Gamma", "gamma_url": None, "file_path": None}
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
    # Inject API key into HTML so the frontend can authenticate requests
    return HTML_PAGE.replace("{{APP_API_KEY}}", APP_API_KEY)


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
        if os.environ.get("GAMMA_API_KEY", ""):
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

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str, request: Request, _=Depends(verify_api_key)):
    # Validate job_id is a UUID to prevent injection
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    info = JOBS.get(job_id)
    if not info or not os.path.exists(info["path"]):
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return HTMLResponse(
                "<html><body style='font-family:sans-serif;text-align:center;padding:4rem;background:#0a0b10;color:#e0e0e8;'>"
                "<h1 style='color:#f87171;'>Download Expired</h1>"
                "<p style='color:#a0a0b8;'>This download link has expired or the file was already cleaned up.</p>"
                "<a href='/' style='color:#7c6aff;'>Generate a new deck</a>"
                "</body></html>",
                status_code=404,
            )
        return JSONResponse({"error": "File not found or expired"}, status_code=404)
    # Path traversal protection
    real_path = os.path.realpath(info["path"])
    if not real_path.startswith(os.path.realpath(TMPDIR)):
        return JSONResponse({"error": "Access denied"}, status_code=403)
    return FileResponse(real_path, filename="presentation.pptx", media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
# Frontend UI
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="api-key" content="{{APP_API_KEY}}">
<title>Multi-Agent Research-to-Deck</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📊</text></svg>">
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
  #agentLog { max-height:400px; overflow-y:auto; }
  .elapsed-timer { color:var(--text-dim); font-size:0.8rem; float:right; }
  .agent-row { display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0;
    border-bottom:1px solid var(--border); font-size:0.82rem; }
  .agent-row:last-child { border-bottom:none; }
  .agent-dot { width:18px; height:18px; border-radius:50%; background:var(--border); flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:0.65rem; color:#fff; }
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
  .collapse-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); border:0; }

  /* Progress steps */
  .progress-steps { display:flex; gap:0; margin-bottom:1rem; border-radius:8px; overflow:hidden; }
  .step { flex:1; text-align:center; padding:0.5rem 0.25rem; font-size:0.7rem; font-weight:600;
    background:var(--surface2); color:var(--text-dim); border-right:1px solid var(--border);
    transition:background 0.3s,color 0.3s; }
  .step:last-child { border-right:none; }
  .step.active { background:var(--accent); color:#fff; }
  .step.done { background:var(--green); color:#fff; }

  @media (max-width:600px) {
    .row3 { grid-template-columns:1fr; }
    .row { grid-template-columns:1fr; }
    .step { font-size:0.6rem; padding:0.4rem 0.15rem; }
  }
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
  <form class="form-card" id="formCard" onsubmit="event.preventDefault(); generate();">
    <h2 style="font-size:1rem;color:var(--text);margin-bottom:0.75rem;">Configure Your Deck</h2>
    <label for="topic">Research Topic</label>
    <input type="text" id="topic" required aria-required="true" aria-describedby="topicError" maxlength="500" placeholder="e.g. The future of AI agents in enterprise software">
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
        <div><label for="audienceRole">Role</label><input type="text" id="audienceRole" maxlength="200" placeholder="e.g. VP Engineering"></div>
        <div><label for="audienceFamiliarity">Familiarity</label>
          <select id="audienceFamiliarity">
            <option value="none">None</option><option value="some" selected>Some</option>
            <option value="expert">Expert</option>
          </select></div>
      </div>
      <label for="audienceMotivation">Motivation</label><input type="text" id="audienceMotivation" maxlength="300" placeholder="What brought them to this meeting?">
      <label for="audienceObjections">Likely Objections</label><input type="text" id="audienceObjections" maxlength="300" placeholder="What will they push back on?">
      <label for="desiredAction">Desired Action</label><input type="text" id="desiredAction" maxlength="300" placeholder="What should they do after?">
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

    <button type="submit" class="btn" id="genBtn">Generate Deck</button>
    <button type="button" class="btn" id="cancelBtn" style="display:none;background:var(--red);margin-top:0.5rem;" onclick="cancelGeneration()">Cancel</button>
  </form>

  <div class="pipeline" id="pipeline" role="region" aria-label="Pipeline progress">
    <div class="progress-steps" id="progressSteps" aria-label="Pipeline steps">
      <span class="step" id="step-research">Research</span>
      <span class="step" id="step-brief">Brief</span>
      <span class="step" id="step-architect">Outline</span>
      <span class="step" id="step-writer">Write</span>
      <span class="step" id="step-reviewer">Review</span>
      <span class="step" id="step-gamma">Design</span>
    </div>
    <h2>Agent Pipeline <span class="elapsed-timer" id="elapsedTimer"></span></h2>
    <p id="progressLabel" style="font-size:0.8rem;color:var(--text-dim);margin-bottom:0.5rem;"></p>
    <div id="agentLog" role="log" aria-live="polite" aria-label="Agent status updates"></div>
    <div id="statusAnnounce" role="status" class="sr-only" aria-live="polite"></div>
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

function statusIcon(s) {
  var icons = {done:'\\u2713', error:'\\u2717', running:'\\u27F3', rejected:'\\u26A0', validated:'\\u2713'};
  return icons[s] || '\\u25CF';
}

function addAgentRow(agent, msg, status) {
  var log = document.getElementById('agentLog');
  var id = 'row-' + agent + '-' + Date.now();
  var row = document.createElement('div');
  row.className = 'agent-row';
  row.id = id;
  row.innerHTML = '<div class="agent-dot '+status+'" aria-hidden="true">'+statusIcon(status)+'</div><span class="sr-only">'+esc(status)+'</span><span class="agent-label">'+esc(agent)+'</span><span class="agent-msg">'+esc(msg)+'</span>';
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return id;
}

function updateLastRow(status, msg) {
  var rows = document.querySelectorAll('.agent-row');
  if (!rows.length) return;
  var last = rows[rows.length - 1];
  var dot = last.querySelector('.agent-dot');
  dot.className = 'agent-dot ' + status;
  dot.textContent = statusIcon(status);
  if (msg) last.querySelector('.agent-msg').textContent = msg;
}

var controller;
var elapsedInterval;
var stepCount = 0;
var TOTAL_STEPS = 6;

function startTimer() {
  var start = Date.now();
  var el = document.getElementById('elapsedTimer');
  elapsedInterval = setInterval(function() {
    var s = Math.floor((Date.now() - start) / 1000);
    el.textContent = Math.floor(s/60) + ':' + String(s%60).padStart(2,'0');
  }, 1000);
}

function stopTimer() {
  if (elapsedInterval) clearInterval(elapsedInterval);
  elapsedInterval = null;
}

function updateProgressLabel(agentName) {
  stepCount++;
  var el = document.getElementById('progressLabel');
  if (el) el.textContent = 'Step ' + stepCount + ' of ' + TOTAL_STEPS + ': ' + agentName;
}

var apiKey = '';

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

  if (!apiKey) {
    var meta = document.querySelector('meta[name="api-key"]');
    apiKey = meta ? meta.content : '';
  }

  var btn = document.getElementById('genBtn');
  var pipeline = document.getElementById('pipeline');
  var result = document.getElementById('result');

  controller = new AbortController();
  btn.disabled = true;
  btn.textContent = 'Agents running...';
  document.getElementById('cancelBtn').style.display = 'block';
  pipeline.classList.add('active');
  result.classList.remove('active'); result.innerHTML = '';
  document.getElementById('agentLog').innerHTML = '';
  document.getElementById('elapsedTimer').textContent = '';
  document.getElementById('progressLabel').textContent = '';
  stepCount = 0;
  resetSteps();
  startTimer();

  addAgentRow('system', 'Connecting to pipeline...', 'running');

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
      headers: {'Content-Type':'application/json', 'X-API-Key': apiKey},
      body: JSON.stringify(body),
      signal: controller.signal
    });

    if (!res.ok) {
      var errData = await res.json().catch(function() { return {}; });
      throw new Error(errData.detail || 'Server error ' + res.status);
    }

    updateLastRow('done', 'Connected');

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
    // Process any remaining data in the buffer after stream ends
    if (buffer.trim()) {
      var remaining = buffer.trim();
      if (remaining.startsWith('data: ')) {
        try { handleEvent(JSON.parse(remaining.slice(6))); } catch(e) {}
      }
    }
  } catch(e) {
    if (e.name !== 'AbortError') {
      result.classList.add('active');
      result.innerHTML = '<p style="color:var(--red)">'+esc(e.message || 'Network error')+'</p>';
    }
  }
  stopTimer();
  btn.disabled = false; btn.textContent = 'Generate Deck';
  document.getElementById('cancelBtn').style.display = 'none';
  controller = null;
}

function cancelGeneration() {
  if (controller) controller.abort();
  stopTimer();
  document.getElementById('cancelBtn').style.display = 'none';
  var btn = document.getElementById('genBtn');
  btn.disabled = false;
  btn.textContent = 'Generate Deck';
  addAgentRow('system', 'Generation cancelled by user', 'error');
  announce('Generation cancelled');
}

var STEP_MAP = {researcher:'step-research', brief:'step-brief', architect:'step-architect', writer:'step-writer', reviewer:'step-reviewer', gamma:'step-gamma'};

function setStepActive(agent) {
  var stepId = STEP_MAP[agent];
  if (!stepId) return;
  // Mark previous steps done
  var steps = document.querySelectorAll('.step');
  var found = false;
  for (var i = 0; i < steps.length; i++) {
    if (steps[i].id === stepId) { found = true; steps[i].className = 'step active'; }
    else if (!found) { steps[i].className = 'step done'; }
  }
}

function setStepDone(agent) {
  var stepId = STEP_MAP[agent];
  if (!stepId) return;
  var el = document.getElementById(stepId);
  if (el) el.className = 'step done';
}

function resetSteps() {
  var steps = document.querySelectorAll('.step');
  for (var i = 0; i < steps.length; i++) steps[i].className = 'step';
}

function announce(msg) {
  var el = document.getElementById('statusAnnounce');
  if (el) el.textContent = msg;
}

function handleEvent(e) {
  switch(e.event) {
    case 'agent_start':
      addAgentRow(e.agent || 'system', e.message, 'running');
      setStepActive(e.agent);
      updateProgressLabel(e.agent || 'system');
      announce(e.message);
      break;
    case 'agent_complete':
      updateLastRow('done', e.message);
      setStepDone(e.agent);
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
      announce('Pipeline complete');
      break;
    case 'complete': {
      var r = document.getElementById('result');
      r.classList.add('active');
      r.setAttribute('tabindex', '-1');
      r.focus();
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
      if (msg) { msg.style.display = 'inline'; setTimeout(function() { msg.style.display = 'none'; }, 3500); }
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

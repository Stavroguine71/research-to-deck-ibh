"""
ORCHESTRATOR — Multi-Agent Research-to-Deck Pipeline
=====================================================
Manages the full pipeline with validation gates between passes.

Architecture:
  1. Researcher Agent — 4 Tavily queries in PARALLEL
  2. Brief Agent → Validator (can reject → retry)
  3. Architect Agent → Validator (can reject → retry)
  4. Writer Agent → Validator (can reject → retry)
  5. Reviewer Agent — final quality gate with rewrites
  6. Output → Gamma API for design

Each agent runs independently with its own Claude API call.
The Validator sits between passes and can reject bad output,
triggering a retry. Max 1 retry per pass to stay within time limits.
"""

import json
import time
from typing import AsyncGenerator
from . import (
    ResearcherAgent,
    BriefAgent,
    ArchitectAgent,
    WriterAgent,
    ReviewerAgent,
    ValidatorAgent,
)

researcher = ResearcherAgent()
brief_agent = BriefAgent()
architect_agent = ArchitectAgent()
writer_agent = WriterAgent()
reviewer_agent = ReviewerAgent()
validator = ValidatorAgent()

MAX_RETRIES = 1  # Max retries per pass (keep tight for Vercel 300s limit)


async def run_pipeline(
    topic: str,
    audience_context: str = "",
    narrative: str = "pir",
    num_slides: int = 10,
    tone: str = "authoritative",
    depth: str = "standard",
) -> AsyncGenerator[dict, None]:
    """
    Run the full multi-agent pipeline with streaming status events.
    Yields events as each agent completes.
    """
    pipeline_start = time.time()

    # ===== PHASE 1: PARALLEL RESEARCH =====
    yield {
        "event": "agent_start",
        "agent": "researcher",
        "message": "Research Agent: Running 4 search queries in parallel...",
    }

    try:
        research_data = await researcher.run(topic)
        yield {
            "event": "agent_complete",
            "agent": "researcher",
            "message": f"Research Agent done — {research_data['total_results']} results from {research_data['queries_succeeded']}/4 queries",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "researcher", "message": str(e)}
        return

    # ===== PHASE 2: BRIEF (Pass 0) + Validation =====
    brief = None
    for attempt in range(1 + MAX_RETRIES):
        label = " (retry)" if attempt > 0 else ""
        yield {
            "event": "agent_start",
            "agent": "brief",
            "message": f"Brief Agent{label}: Synthesizing research into structured brief...",
        }
        try:
            brief = await brief_agent.run(research_data, audience_context)
            yield {
                "event": "agent_complete",
                "agent": "brief",
                "message": f"Brief Agent done — {len(brief.get('findings', []))} findings, confidence: {brief.get('confidence', '?')}",
            }
        except Exception as e:
            yield {"event": "agent_error", "agent": "brief", "message": str(e)}
            if attempt == MAX_RETRIES:
                return
            continue

        # Validate
        yield {"event": "validating", "agent": "validator", "message": "Validator checking brief quality..."}
        try:
            v = await validator.run("brief", brief, expected_by="architect")
            if v["verdict"] == "pass":
                yield {"event": "validated", "agent": "validator", "message": f"Brief validated — score {v['score']}/10"}
                break
            else:
                yield {
                    "event": "rejected",
                    "agent": "validator",
                    "message": f"Brief rejected (score {v['score']}/10): {'; '.join(v.get('issues', [])[:2])}",
                }
                if attempt == MAX_RETRIES:
                    yield {"event": "warning", "message": "Proceeding with current brief despite validation failure"}
                    break
        except Exception:
            break  # Validator failure shouldn't block the pipeline

    if brief is None:
        return

    # ===== PHASE 3: ARCHITECT (Pass 1) + Validation =====
    outline = None
    for attempt in range(1 + MAX_RETRIES):
        label = " (retry)" if attempt > 0 else ""
        yield {
            "event": "agent_start",
            "agent": "architect",
            "message": f"Architect Agent{label}: Designing {num_slides}-slide deck with {narrative.upper()} narrative...",
        }
        try:
            outline = await architect_agent.run(
                brief, audience_context, narrative, num_slides, tone, depth
            )
            slide_count = len(outline.get("slides", []))
            yield {
                "event": "agent_complete",
                "agent": "architect",
                "message": f"Architect done — {slide_count} slides designed",
            }
        except Exception as e:
            yield {"event": "agent_error", "agent": "architect", "message": str(e)}
            if attempt == MAX_RETRIES:
                return
            continue

        # Validate
        yield {"event": "validating", "agent": "validator", "message": "Validator checking outline quality..."}
        try:
            v = await validator.run("architect", outline, expected_by="writer")
            if v["verdict"] == "pass":
                yield {"event": "validated", "agent": "validator", "message": f"Outline validated — score {v['score']}/10"}
                break
            else:
                yield {
                    "event": "rejected",
                    "agent": "validator",
                    "message": f"Outline rejected (score {v['score']}/10): {'; '.join(v.get('issues', [])[:2])}",
                }
                if attempt == MAX_RETRIES:
                    yield {"event": "warning", "message": "Proceeding with current outline despite validation failure"}
                    break
        except Exception:
            break

    if outline is None:
        return

    # ===== PHASE 4: WRITER (Pass 2) + Validation =====
    content = None
    for attempt in range(1 + MAX_RETRIES):
        label = " (retry)" if attempt > 0 else ""
        yield {
            "event": "agent_start",
            "agent": "writer",
            "message": f"Writer Agent{label}: Writing full slide content with data and speaker notes...",
        }
        try:
            content = await writer_agent.run(outline, brief, audience_context)
            yield {
                "event": "agent_complete",
                "agent": "writer",
                "message": f"Writer done — {len(content.get('slides', []))} slides written",
            }
        except Exception as e:
            yield {"event": "agent_error", "agent": "writer", "message": str(e)}
            if attempt == MAX_RETRIES:
                return
            continue

        # Validate
        yield {"event": "validating", "agent": "validator", "message": "Validator checking content quality..."}
        try:
            v = await validator.run("writer", content, expected_by="reviewer")
            if v["verdict"] == "pass":
                yield {"event": "validated", "agent": "validator", "message": f"Content validated — score {v['score']}/10"}
                break
            else:
                yield {
                    "event": "rejected",
                    "agent": "validator",
                    "message": f"Content rejected (score {v['score']}/10): {'; '.join(v.get('issues', [])[:2])}",
                }
                if attempt == MAX_RETRIES:
                    yield {"event": "warning", "message": "Proceeding with current content despite validation failure"}
                    break
        except Exception:
            break

    if content is None:
        return

    # ===== PHASE 5: REVIEWER (Pass 3) — Final Quality Gate =====
    yield {
        "event": "agent_start",
        "agent": "reviewer",
        "message": "Senior Partner reviewing all slides — scoring and rewriting weak ones...",
    }
    try:
        review = await reviewer_agent.run(content, brief)
        rewritten = review.get("slides_rewritten", 0)
        overall = review.get("overall_score", "?")
        yield {
            "event": "agent_complete",
            "agent": "reviewer",
            "message": f"Partner Review done — score {overall}/10, {rewritten} slides rewritten",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "reviewer", "message": str(e)}
        # Fall back to unreviewed content
        review = content
        review["overall_score"] = "N/A"

    total_elapsed = round(time.time() - pipeline_start, 1)

    # Build the final deck plan (compatible with Gamma formatting)
    final_plan = {
        "title": topic,
        "story_spine": outline.get("story_spine", ""),
        "overall_score": review.get("overall_score", "?"),
        "narrative_coherence": review.get("narrative_coherence", ""),
        "slides": review.get("slides", content.get("slides", [])),
        "counterarguments_addressed": review.get("counterarguments_addressed", False),
        "actionable_ask_present": review.get("actionable_ask_present", False),
        "weakest_dimension": review.get("weakest_dimension", ""),
        "_pipeline": {
            "total_elapsed_seconds": total_elapsed,
            "agents_ran": 6,
            "research_results": research_data["total_results"],
            "slides_rewritten": review.get("slides_rewritten", 0),
            "validation_passes": 3,
        },
    }

    yield {
        "event": "pipeline_complete",
        "message": f"Multi-agent pipeline complete in {total_elapsed}s",
        "deck_plan": final_plan,
    }

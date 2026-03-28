"""
ORCHESTRATOR — Multi-Agent Research-to-Deck Pipeline
=====================================================
Optimized for Vercel's 300s function timeout.

Pipeline (~180s total):
  1. Researcher — 4 parallel Tavily queries (~5s)
  2. Brief Agent (~20s) → Validator (~8s)
  3. Architect Agent (~20s)
  4. Writer Agent (~40s)
  5. Reviewer Agent (~50s)
  6. Gamma API (~30s)

No retries — each agent gets one shot to stay within timeout.
Validator only on the Brief pass (most impactful gate).
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
    Optimized to complete within Vercel's 300s timeout.
    """
    pipeline_start = time.time()

    def elapsed():
        return round(time.time() - pipeline_start, 1)

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
            "message": f"Research done — {research_data['total_results']} results ({elapsed()}s)",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "researcher", "message": str(e)}
        return

    # ===== PHASE 2: BRIEF + Validation =====
    yield {
        "event": "agent_start",
        "agent": "brief",
        "message": "Brief Agent: Synthesizing research into structured brief...",
    }
    try:
        brief = await brief_agent.run(research_data, audience_context)
        yield {
            "event": "agent_complete",
            "agent": "brief",
            "message": f"Brief done — {len(brief.get('findings', []))} findings ({elapsed()}s)",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "brief", "message": str(e)}
        return

    # Validate the brief — most impactful gate (garbage in = garbage out)
    yield {"event": "validating", "agent": "validator", "message": "Validator checking brief quality..."}
    try:
        v = await validator.run("brief", brief, expected_by="architect")
        if v["verdict"] == "pass":
            yield {"event": "validated", "agent": "validator", "message": f"Brief validated — score {v['score']}/10 ({elapsed()}s)"}
        else:
            yield {
                "event": "rejected",
                "agent": "validator",
                "message": f"Brief weak (score {v['score']}/10) — proceeding anyway ({elapsed()}s)",
            }
    except Exception:
        yield {"event": "validated", "agent": "validator", "message": f"Validator skipped ({elapsed()}s)"}

    # ===== PHASE 3: ARCHITECT =====
    yield {
        "event": "agent_start",
        "agent": "architect",
        "message": f"Architect: Designing {num_slides}-slide deck with {narrative.upper()} narrative...",
    }
    try:
        outline = await architect_agent.run(
            brief, audience_context, narrative, num_slides, tone, depth
        )
        slide_count = len(outline.get("slides", []))
        yield {
            "event": "agent_complete",
            "agent": "architect",
            "message": f"Architect done — {slide_count} slides designed ({elapsed()}s)",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "architect", "message": str(e)}
        return

    # ===== PHASE 4: WRITER =====
    yield {
        "event": "agent_start",
        "agent": "writer",
        "message": "Writer: Filling slides with data-backed content and speaker notes...",
    }
    try:
        content = await writer_agent.run(outline, brief, audience_context)
        yield {
            "event": "agent_complete",
            "agent": "writer",
            "message": f"Writer done — {len(content.get('slides', []))} slides written ({elapsed()}s)",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "writer", "message": str(e)}
        return

    # ===== PHASE 5: REVIEWER — Final Quality Gate =====
    yield {
        "event": "agent_start",
        "agent": "reviewer",
        "message": "Senior Partner reviewing — scoring and rewriting weak slides...",
    }
    try:
        review = await reviewer_agent.run(content, brief)
        rewritten = review.get("slides_rewritten", 0)
        overall = review.get("overall_score", "?")
        yield {
            "event": "agent_complete",
            "agent": "reviewer",
            "message": f"Review done — score {overall}/10, {rewritten} rewritten ({elapsed()}s)",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "reviewer", "message": str(e)}
        # Fall back to unreviewed content
        review = content
        review["overall_score"] = "N/A"

    total_elapsed = round(time.time() - pipeline_start, 1)

    # Build the final deck plan
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
        },
    }

    yield {
        "event": "pipeline_complete",
        "message": f"Pipeline complete in {total_elapsed}s",
        "deck_plan": final_plan,
    }

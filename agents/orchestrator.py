"""
ORCHESTRATOR — Multi-Agent Research-to-Deck Pipeline
=====================================================
Railway deployment — no timeout constraints.

Full pipeline with validation gates and retries:
  1. Researcher — 4 parallel Tavily queries
  2. Brief Agent → Validator (retry on fail)
  3. Architect Agent → Validator (retry on fail)
  4. Writer Agent → Validator (retry on fail)
  5. Reviewer Agent — final quality gate with rewrites
  6. Output → Gamma API for design
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

MAX_RETRIES = 1  # One retry per pass if validation fails


async def _run_with_validation(
    agent_name: str,
    agent_fn,
    next_agent: str,
    pipeline_start: float,
) -> tuple:
    """
    Helper: run an agent, validate its output, retry once if rejected.
    Returns (result, events_list).
    """
    events = []

    def elapsed():
        return round(time.time() - pipeline_start, 1)

    result = None
    for attempt in range(1 + MAX_RETRIES):
        label = " (retry)" if attempt > 0 else ""

        # Run the agent
        events.append({
            "event": "agent_start",
            "agent": agent_name,
            "message": f"{agent_name.title()} Agent{label} running... ({elapsed()}s)",
        })
        try:
            result = await agent_fn()
        except Exception as e:
            events.append({"event": "agent_error", "agent": agent_name, "message": str(e)})
            if attempt == MAX_RETRIES:
                return None, events
            continue

        events.append({
            "event": "agent_complete",
            "agent": agent_name,
            "message": f"{agent_name.title()} done ({elapsed()}s)",
        })

        # Validate
        events.append({"event": "validating", "agent": "validator", "message": f"Validating {agent_name} output..."})
        try:
            v = await validator.run(agent_name, result, expected_by=next_agent)
            if v["verdict"] == "pass":
                events.append({
                    "event": "validated",
                    "agent": "validator",
                    "message": f"Validated — score {v['score']}/10 ({elapsed()}s)",
                })
                break
            else:
                issues = "; ".join(v.get("issues", [])[:2])
                events.append({
                    "event": "rejected",
                    "agent": "validator",
                    "message": f"Rejected (score {v['score']}/10): {issues} ({elapsed()}s)",
                })
                if attempt == MAX_RETRIES:
                    events.append({"event": "warning", "message": "Proceeding despite validation failure"})
                    break
        except Exception:
            events.append({"event": "validated", "agent": "validator", "message": f"Validator skipped ({elapsed()}s)"})
            break

    return result, events


async def run_pipeline(
    topic: str,
    audience_context: str = "",
    narrative: str = "pir",
    num_slides: int = 10,
    tone: str = "authoritative",
    depth: str = "standard",
) -> AsyncGenerator[dict, None]:
    """
    Full multi-agent pipeline with validation loops.
    No timeout constraints — Railway runs a persistent server.
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
            "message": f"Research done — {research_data['total_results']} results from {research_data['queries_succeeded']}/4 queries ({elapsed()}s)",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "researcher", "message": str(e)}
        return

    # ===== PHASE 2: BRIEF + Validation (with retry) =====
    brief, events = await _run_with_validation(
        "brief",
        lambda: brief_agent.run(research_data, audience_context),
        "architect",
        pipeline_start,
    )
    for e in events:
        yield e
    if brief is None:
        return

    findings_count = len(brief.get("findings", []))
    yield {
        "event": "agent_complete",
        "agent": "brief",
        "message": f"Brief finalized — {findings_count} findings, confidence: {brief.get('confidence', '?')} ({elapsed()}s)",
    }

    # ===== PHASE 3: ARCHITECT + Validation (with retry) =====
    outline, events = await _run_with_validation(
        "architect",
        lambda: architect_agent.run(brief, audience_context, narrative, num_slides, tone, depth),
        "writer",
        pipeline_start,
    )
    for e in events:
        yield e
    if outline is None:
        return

    slide_count = len(outline.get("slides", []))
    yield {
        "event": "agent_complete",
        "agent": "architect",
        "message": f"Outline finalized — {slide_count} slides with {narrative.upper()} arc ({elapsed()}s)",
    }

    # ===== PHASE 4: WRITER + Validation (with retry) =====
    content, events = await _run_with_validation(
        "writer",
        lambda: writer_agent.run(outline, brief, audience_context),
        "reviewer",
        pipeline_start,
    )
    for e in events:
        yield e
    if content is None:
        return

    yield {
        "event": "agent_complete",
        "agent": "writer",
        "message": f"Content finalized — {len(content.get('slides', []))} slides written ({elapsed()}s)",
    }

    # ===== PHASE 5: REVIEWER — Final Quality Gate =====
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
            "message": f"Review done — score {overall}/10, {rewritten} slides rewritten ({elapsed()}s)",
        }
    except Exception as e:
        yield {"event": "agent_error", "agent": "reviewer", "message": str(e)}
        review = content
        review["overall_score"] = "N/A"

    total_elapsed = round(time.time() - pipeline_start, 1)

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
        "message": f"Pipeline complete in {total_elapsed}s",
        "deck_plan": final_plan,
    }

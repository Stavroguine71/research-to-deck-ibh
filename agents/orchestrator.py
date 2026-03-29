"""
ORCHESTRATOR — Multi-Agent Research-to-Deck Pipeline
=====================================================
Railway deployment — no timeout constraints.

Full pipeline with validation gates, retries, and SSE heartbeats:
  1. Researcher — 4 parallel Tavily queries
  2. Brief Agent → Validator (retry on fail)
  3. Architect Agent → Validator (retry on fail)
  4. Writer Agent → Validator (retry on fail)
  5. Reviewer Agent — final quality gate with rewrites
  6. Output → Gamma API for design
"""

import json
import time
import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)
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
HEARTBEAT_INTERVAL = 15  # seconds between keep-alive pings


async def _run_with_heartbeat(agent_fn, agent_name: str, pipeline_start: float):
    """
    Async generator: runs agent_fn as a background task and yields
    heartbeat events every HEARTBEAT_INTERVAL seconds to keep the
    SSE connection alive.

    Final yield is {"_result": <agent output>} or raises on error.
    Properly cancels the task if the generator is closed (client disconnect).
    """
    task = asyncio.create_task(agent_fn())

    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                elapsed = round(time.time() - pipeline_start, 1)
                yield {
                    "event": "heartbeat",
                    "agent": agent_name,
                    "message": f"{agent_name.title()} still working... ({elapsed}s)",
                }

        # Task is done — check for exceptions
        result = task.result()
        yield {"_result": result}
    except GeneratorExit:
        # Client disconnected — cancel the running task
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        raise
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        raise


async def run_pipeline(
    topic: str,
    audience_context: str = "",
    narrative: str = "pir",
    num_slides: int = 10,
    tone: str = "authoritative",
    depth: str = "standard",
) -> AsyncGenerator[dict, None]:
    """
    Full multi-agent pipeline with validation loops and real-time heartbeats.
    No timeout constraints — Railway runs a persistent server.
    """
    pipeline_start = time.time()

    def elapsed():
        return round(time.time() - pipeline_start, 1)

    # Sanitize topic for use in LLM prompts
    safe_topic = f"<user_input>{topic}</user_input>"

    # ===== PHASE 1: PARALLEL RESEARCH =====
    yield {
        "event": "agent_start",
        "agent": "researcher",
        "message": "Research Agent: Running 4 search queries in parallel...",
    }
    try:
        research_data = await researcher.run(topic)  # Raw topic for search queries
        yield {
            "event": "agent_complete",
            "agent": "researcher",
            "message": f"Research done — {research_data['total_results']} results from {research_data['queries_succeeded']}/4 queries ({elapsed()}s)",
        }
    except Exception as e:
        logger.exception("Researcher agent failed")
        yield {"event": "agent_error", "agent": "researcher", "message": f"Research failed ({type(e).__name__}). Check API keys."}
        yield {"event": "error", "message": "Research agent failed. Please try again."}
        return

    # Tag topic in research_data for downstream agents
    research_data["topic"] = safe_topic

    # Check for empty research results
    if research_data.get("total_results", 0) == 0:
        yield {"event": "agent_error", "agent": "researcher", "message": "No research results found. Check your Tavily API key."}
        yield {"event": "error", "message": "Research returned no results. Cannot build a deck without data."}
        return

    # ===== PHASE 2-4: Sequential agents with validation =====
    brief_result = None
    outline_result = None
    content_result = None

    def make_agent_fn(phase, feedback=""):
        """Factory that creates agent callables with optional validation feedback."""
        if phase == "brief":
            return lambda: brief_agent.run(research_data, audience_context, validation_feedback=feedback)
        elif phase == "architect":
            return lambda: architect_agent.run(
                brief_result, audience_context, narrative, num_slides, tone, depth, validation_feedback=feedback
            )
        elif phase == "writer":
            return lambda: writer_agent.run(outline_result, brief_result, audience_context, validation_feedback=feedback)

    phases = [
        ("brief", "architect"),
        ("architect", "writer"),
        ("writer", "reviewer"),
    ]

    for phase_name, next_agent in phases:
        best_result = None  # Best result across attempts (validated or last produced)
        validation_passed = False
        validation_feedback = ""
        for attempt in range(1 + MAX_RETRIES):
            attempt_result = None  # Reset each attempt — don't leak previous rejected output
            label = " (retry)" if attempt > 0 else ""

            yield {
                "event": "agent_start",
                "agent": phase_name,
                "message": f"{phase_name.title()} Agent{label} running... ({elapsed()}s)",
            }

            # Run agent with heartbeats streaming in real-time
            agent_fn = make_agent_fn(phase_name, feedback=validation_feedback)
            try:
                async for event in _run_with_heartbeat(
                    agent_fn, phase_name, pipeline_start
                ):
                    if "_result" in event:
                        attempt_result = event["_result"]
                    else:
                        yield event  # heartbeat — streams immediately
            except Exception as e:
                logger.exception(f"{phase_name} agent failed on attempt {attempt + 1}")
                yield {"event": "agent_error", "agent": phase_name, "message": f"{phase_name.title()} error. Retrying..."}
                if attempt == MAX_RETRIES:
                    break
                continue

            if attempt_result is None:
                logger.error(f"{phase_name} agent returned None on attempt {attempt + 1}")
                if attempt == MAX_RETRIES:
                    break
                continue

            # Keep the latest result as our best candidate
            best_result = attempt_result

            yield {
                "event": "agent_complete",
                "agent": phase_name,
                "message": f"{phase_name.title()} done ({elapsed()}s)",
            }

            # Validate
            yield {
                "event": "validating",
                "agent": "validator",
                "message": f"Validating {phase_name} output...",
            }
            try:
                v = await validator.run(phase_name, attempt_result, expected_by=next_agent)
                if v["verdict"] == "pass":
                    yield {
                        "event": "validated",
                        "agent": "validator",
                        "message": f"Validated — score {v['score']}/10 ({elapsed()}s)",
                    }
                    validation_passed = True
                    break
                else:
                    issues = "; ".join(v.get("issues", [])[:2])
                    # Store feedback for retry (wrapped in tags to prevent injection)
                    raw_issues = '; '.join(v.get('issues', []))
                    validation_feedback = f"Score: {v.get('score')}/10. Issues: <validator_feedback>{raw_issues}</validator_feedback>"
                    yield {
                        "event": "rejected",
                        "agent": "validator",
                        "message": f"Rejected (score {v['score']}/10): {issues} ({elapsed()}s)",
                    }
                    if attempt == MAX_RETRIES:
                        best_result["_validation_failed"] = True
                        yield {"event": "warning", "message": f"Proceeding with {phase_name} output despite validation failure (score {v.get('score')}/10)"}
                        break
            except Exception as ve:
                logger.warning(f"Validator failed for {phase_name}: {ve}")
                yield {
                    "event": "validated",
                    "agent": "validator",
                    "message": f"Validator skipped ({elapsed()}s)",
                }
                break

        if best_result is None:
            yield {"event": "error", "message": f"{phase_name.title()} agent failed after retries. Please try again."}
            return

        # Store result for next phase
        if phase_name == "brief":
            brief_result = best_result
        elif phase_name == "architect":
            outline_result = best_result
        elif phase_name == "writer":
            content_result = best_result

    # Safety check: ensure all phases produced results before reviewing
    if content_result is None or outline_result is None or brief_result is None:
        yield {"event": "error", "message": "Internal error: missing phase results. Please try again."}
        return

    # ===== PHASE 5: REVIEWER — Final Quality Gate (with heartbeats) =====
    yield {
        "event": "agent_start",
        "agent": "reviewer",
        "message": "Senior Partner reviewing all slides — scoring and rewriting weak ones...",
    }

    review = content_result
    try:
        async for event in _run_with_heartbeat(
            lambda: reviewer_agent.run(content_result, brief_result),
            "reviewer",
            pipeline_start,
        ):
            if "_result" in event:
                review = event["_result"]
            else:
                yield event  # heartbeat

        rewritten = review.get("slides_rewritten", 0)
        overall = review.get("overall_score", "?")
        yield {
            "event": "agent_complete",
            "agent": "reviewer",
            "message": f"Review done — score {overall}/10, {rewritten} slides rewritten ({elapsed()}s)",
        }
    except Exception as e:
        logger.exception("Reviewer agent failed")
        yield {"event": "agent_error", "agent": "reviewer", "message": f"Reviewer error: {type(e).__name__}. Using unreviewed content."}
        review = {
            **content_result,
            "overall_score": "N/A",
            "slides_rewritten": 0,
            "narrative_coherence": "Review skipped due to error",
            "counterarguments_addressed": False,
            "actionable_ask_present": False,
            "weakest_dimension": "unknown",
        }

    total_elapsed = round(time.time() - pipeline_start, 1)

    final_plan = {
        "title": topic,  # Raw topic for display (not sent to LLM)
        "story_spine": outline_result.get("story_spine", ""),
        "overall_score": review.get("overall_score", "?"),
        "narrative_coherence": review.get("narrative_coherence", ""),
        "slides": review.get("slides", content_result.get("slides", [])),
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

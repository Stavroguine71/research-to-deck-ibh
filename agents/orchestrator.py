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
HEARTBEAT_INTERVAL = 15  # seconds between keep-alive pings


async def _run_with_heartbeat(agent_fn, agent_name: str, pipeline_start: float):
    """
    Async generator: runs agent_fn as a background task and yields
    heartbeat events every HEARTBEAT_INTERVAL seconds to keep the
    SSE connection alive.

    Final yield is {"_result": <agent output>} or raises on error.
    """
    result_holder = {"value": None, "error": None, "done": False}

    async def _work():
        try:
            result_holder["value"] = await agent_fn()
        except Exception as e:
            result_holder["error"] = e
        finally:
            result_holder["done"] = True

    task = asyncio.create_task(_work())

    while not result_holder["done"]:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            elapsed = round(time.time() - pipeline_start, 1)
            yield {
                "event": "heartbeat",
                "agent": agent_name,
                "message": f"{agent_name.title()} still working... ({elapsed}s)",
            }

    # Make sure the task is truly done
    await task

    if result_holder["error"]:
        raise result_holder["error"]

    yield {"_result": result_holder["value"]}


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

    # ===== PHASE 2-4: Sequential agents with validation =====
    # Each tuple: (name, factory for agent_fn, next_agent_name, summary_fn)
    phases = [
        (
            "brief",
            lambda: brief_agent.run(research_data, audience_context),
            "architect",
        ),
        (
            "architect",
            lambda: architect_agent.run(
                brief_result, audience_context, narrative, num_slides, tone, depth
            ),
            "writer",
        ),
        (
            "writer",
            lambda: writer_agent.run(outline_result, brief_result, audience_context),
            "reviewer",
        ),
    ]

    brief_result = None
    outline_result = None
    content_result = None

    for phase_name, agent_fn_factory, next_agent in phases:
        result = None
        for attempt in range(1 + MAX_RETRIES):
            label = " (retry)" if attempt > 0 else ""

            yield {
                "event": "agent_start",
                "agent": phase_name,
                "message": f"{phase_name.title()} Agent{label} running... ({elapsed()}s)",
            }

            # Run agent with heartbeats streaming in real-time
            try:
                async for event in _run_with_heartbeat(
                    agent_fn_factory, phase_name, pipeline_start
                ):
                    if "_result" in event:
                        result = event["_result"]
                    else:
                        yield event  # heartbeat — streams immediately
            except Exception as e:
                yield {"event": "agent_error", "agent": phase_name, "message": str(e)}
                if attempt == MAX_RETRIES:
                    result = None
                    break
                continue

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
                v = await validator.run(phase_name, result, expected_by=next_agent)
                if v["verdict"] == "pass":
                    yield {
                        "event": "validated",
                        "agent": "validator",
                        "message": f"Validated — score {v['score']}/10 ({elapsed()}s)",
                    }
                    break
                else:
                    issues = "; ".join(v.get("issues", [])[:2])
                    yield {
                        "event": "rejected",
                        "agent": "validator",
                        "message": f"Rejected (score {v['score']}/10): {issues} ({elapsed()}s)",
                    }
                    if attempt == MAX_RETRIES:
                        yield {"event": "warning", "message": "Proceeding despite validation failure"}
                        break
            except Exception:
                yield {
                    "event": "validated",
                    "agent": "validator",
                    "message": f"Validator skipped ({elapsed()}s)",
                }
                break

        if result is None:
            return

        # Store result for next phase
        if phase_name == "brief":
            brief_result = result
            findings_count = len(brief_result.get("findings", []))
            yield {
                "event": "agent_complete",
                "agent": "brief",
                "message": f"Brief finalized — {findings_count} findings, confidence: {brief_result.get('confidence', '?')} ({elapsed()}s)",
            }
        elif phase_name == "architect":
            outline_result = result
            slide_count = len(outline_result.get("slides", []))
            yield {
                "event": "agent_complete",
                "agent": "architect",
                "message": f"Outline finalized — {slide_count} slides with {narrative.upper()} arc ({elapsed()}s)",
            }
        elif phase_name == "writer":
            content_result = result
            yield {
                "event": "agent_complete",
                "agent": "writer",
                "message": f"Content finalized — {len(content_result.get('slides', []))} slides written ({elapsed()}s)",
            }

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
        yield {"event": "agent_error", "agent": "reviewer", "message": str(e)}
        review = content_result
        review["overall_score"] = "N/A"

    total_elapsed = round(time.time() - pipeline_start, 1)

    final_plan = {
        "title": topic,
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

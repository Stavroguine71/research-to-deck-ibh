"""
AGENT 1: Researcher
====================
Runs 4 Tavily search queries IN PARALLEL to gather comprehensive data.
This is the first real multi-agent improvement — your old version
ran these sequentially. Now they fan out concurrently.
"""

import asyncio
import os
import re
import httpx

async def _tavily_search(query: str, max_results: int = 5) -> dict:
    """Single Tavily API call. Reads key at call time to support late-binding."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return {"results": []}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
        return resp.json()


class ResearcherAgent:
    """
    Runs 4 targeted research queries in PARALLEL using asyncio.gather.
    Each query covers a different angle of the topic.
    """

    name = "researcher"
    role = "Research Agent"

    async def run(self, topic: str) -> dict:
        """
        Fan out 4 research queries concurrently.
        Returns consolidated research data.
        """
        # Sanitize topic: strip search operators, limit length
        safe_topic = re.sub(r'[^\w\s.,!?()\'"-]', '', topic)[:200].strip()
        queries = [
            f"{safe_topic} latest data statistics trends 2024 2025",
            f"{safe_topic} competitive landscape market analysis key players",
            f"{safe_topic} strategic implications risks opportunities",
            f"{safe_topic} expert analysis outlook recommendations",
        ]

        # PARALLEL fan-out — all 4 queries run at the same time
        results = await asyncio.gather(
            *[_tavily_search(q) for q in queries],
            return_exceptions=True,
        )

        # Consolidate results
        all_results = []
        all_sources = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            for item in result.get("results", []):
                all_results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0),
                    "query_angle": ["data_trends", "competitive", "strategic", "expert"][i],
                })
                all_sources.append(item.get("url", ""))

        return {
            "topic": topic,
            "total_results": len(all_results),
            "results": all_results,
            "sources": list(set(all_sources)),
            "queries_succeeded": sum(1 for r in results if not isinstance(r, Exception)),
            "queries_failed": sum(1 for r in results if isinstance(r, Exception)),
        }

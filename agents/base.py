"""
Base agent class for the Research-to-Deck pipeline.
Each agent gets its own independent Claude API call.
"""

import json
import time
import httpx
import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = "claude-opus-4-20250514"


async def call_claude(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 6000,
    thinking_budget: int = 8000,
    timeout: float = 600.0,
) -> dict:
    """
    Make an independent Claude API call.
    Each agent call goes through here — its own request, its own context.
    Supports extended thinking for deeper reasoning.
    """
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    if thinking_budget > 0:
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        body["temperature"] = 1  # Required with extended thinking

    start = time.time()

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Retry without thinking if it fails
            if thinking_budget > 0:
                body.pop("thinking", None)
                body.pop("temperature", None)
                headers.pop("anthropic-beta", None)
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
            else:
                raise

    elapsed = round(time.time() - start, 1)

    # Extract text from response (skip thinking blocks)
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text = block["text"]
            break

    return {
        "text": text,
        "elapsed": elapsed,
        "model": model,
        "usage": data.get("usage", {}),
    }


def parse_json_response(text: str) -> dict:
    """Parse JSON from Claude's response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned)

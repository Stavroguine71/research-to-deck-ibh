"""
Base agent class for the Research-to-Deck pipeline.
Each agent gets its own independent Claude API call.
"""

import json
import re
import time
import httpx
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"


def validate_required_keys():
    """Fail fast at startup if required API keys are missing."""
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


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
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
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
        except httpx.HTTPStatusError as e:
            # Log the response body for diagnostics
            try:
                err_body = e.response.text[:500]
            except Exception:
                err_body = "(could not read response body)"
            logger.error(f"Claude API {e.response.status_code}: {err_body}")
            # Only retry without thinking for 400/529 errors
            if thinking_budget > 0 and e.response.status_code in (400, 529):
                logger.warning(f"Claude API {e.response.status_code} with thinking enabled, retrying without thinking")
                body.pop("thinking", None)
                body.pop("temperature", None)
                headers.pop("anthropic-beta", None)
                try:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as retry_err:
                    raise RuntimeError(
                        f"Claude API call failed after retry (status {retry_err.response.status_code})"
                    ) from retry_err
                except Exception as retry_err:
                    raise RuntimeError(
                        "Claude API call failed after retry (network error)"
                    ) from retry_err
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
    """Parse JSON from Claude's response, handling common output variations."""
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    # First try: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Second try: iterative search for valid JSON objects starting at each '{'
    for i, ch in enumerate(cleaned):
        if ch == '{':
            try:
                result = json.loads(cleaned[i:])
                return result
            except json.JSONDecodeError:
                # Try finding the matching closing brace via decoder
                decoder = json.JSONDecoder()
                try:
                    obj, _ = decoder.raw_decode(cleaned, i)
                    return obj
                except json.JSONDecodeError:
                    continue

    # Third try: extract JSON array (log warning — callers should validate structure)
    for i, ch in enumerate(cleaned):
        if ch == '[':
            try:
                arr = json.loads(cleaned[i:])
                logger.warning("parse_json_response: Got JSON array instead of object — wrapping as {items: [...]}")
                return {"items": arr, "_was_array": True}
            except json.JSONDecodeError:
                decoder = json.JSONDecoder()
                try:
                    arr, _ = decoder.raw_decode(cleaned, i)
                    logger.warning("parse_json_response: Got JSON array instead of object — wrapping as {items: [...]}")
                    return {"items": arr, "_was_array": True}
                except json.JSONDecodeError:
                    continue

    logger.error(f"Could not parse JSON from Claude response (first 200 chars): {text[:200]}")
    raise ValueError("Could not parse JSON from Claude response")

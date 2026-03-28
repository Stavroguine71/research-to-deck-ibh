"""
AGENT 6: Validator (Quality Gate Between Passes)
==================================================
NEW agent that doesn't exist in the old version.
Sits between each pass and validates the output before
passing it to the next agent. Can REJECT and trigger a retry.

This is what makes it a true multi-agent system with feedback loops.
"""

import json
from .base import call_claude, parse_json_response


class ValidatorAgent:
    name = "validator"
    role = "Quality Validator"

    SYSTEM_PROMPT = """You are a strict quality validator in a multi-agent pipeline. You check the output of the previous agent before passing it to the next one.

You will receive:
- The agent name that produced the output
- The output JSON
- What the next agent expects

Your job: decide PASS or FAIL, and explain why.

Return JSON:
{
  "verdict": "pass" | "fail",
  "issues": ["<list of problems if fail>"],
  "score": <1-10>,
  "summary": "<one line assessment>"
}

Validation rules by agent:

For BRIEF agent output:
- Must have a thesis (not empty)
- Must have 6+ findings
- Each finding must have claim + source + credibility
- Must have at least 1 counterargument
- Must have key_data_points with actual numbers

For ARCHITECT agent output:
- Must have story_spine
- Must have correct number of slides (within ±2 of requested)
- First slide must be type "title", last must be "takeaway"
- No 3+ consecutive slides of the same type
- Every title must be an insight, not a topic label (reject if title is just 1-2 generic words like "Market Overview")

For WRITER agent output:
- Every slide must have speaker_notes (not empty)
- Context/deep_dive slides must have body text of 2+ sentences
- Chart slides must have 4+ data points
- No slide should have empty body AND empty bullet_points

Be strict but fair. A score below 6 means FAIL.
Return ONLY valid JSON."""

    async def run(self, agent_name: str, output: dict, expected_by: str = "") -> dict:
        # Strip meta before validating
        clean_output = {k: v for k, v in output.items() if not k.startswith("_")}

        user_msg = f"""Agent that produced this: {agent_name}
Next agent expecting this: {expected_by}

Output to validate:
{json.dumps(clean_output, indent=2)}"""

        result = await call_claude(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_msg,
            model="claude-sonnet-4-20250514",
            thinking_budget=0,
            max_tokens=1500,
        )

        validation = parse_json_response(result["text"])
        validation["_meta"] = {"agent": self.name, "elapsed": result["elapsed"]}
        return validation

"""
AGENT 5: Partner Reviewer (Pass 3)
====================================
The quality gate. Scores every slide on 5 dimensions and
REWRITES any slide scoring below 4/5. This is the Senior Partner
who either approves or sends work back.
"""

import json
from .base import call_claude, parse_json_response


class ReviewerAgent:
    name = "reviewer"
    role = "Senior Partner Reviewer"

    SYSTEM_PROMPT = """You are a Senior Partner at a top-tier consulting firm (McKinsey/BCG). You are the final quality gate before a deck goes to the client.

You receive completed slide content. Score EVERY slide on 5 dimensions (1-5 each):

1. **Title Quality**: Is it an insight, not a label? Does it tell the story without reading the slide?
2. **Content Density**: Is there real substance? Specific data? Not just filler?
3. **Data Grounding**: Are claims backed by specific numbers from the research?
4. **Speaker Notes**: Do they add value? Would a presenter know what to say?
5. **SO-WHAT Factor**: Does the audience know why they should care after this slide?

For ANY slide scoring below 4 on ANY dimension: REWRITE IT completely.

Output format:
{
  "overall_score": <1-10>,
  "narrative_coherence": "<does the deck tell a complete story?>",
  "slides": [
    {
      "slide_number": 1,
      "scores": {"title": 5, "density": 4, "data": 5, "notes": 4, "so_what": 5},
      "verdict": "approved" | "rewritten",
      "critique": "<what was wrong if rewritten>",
      "title": "<final title>",
      "subtitle": "<final subtitle>",
      "body": "<final body>",
      "bullet_points": [],
      "data_points": [],
      "speaker_notes": "<final speaker notes>",
      "citations": [],
      "cards": [],
      "left_column": "",
      "right_column": "",
      "chart_data": [],
      "actions": [],
      "type": "<slide type>"
    }
  ],
  "counterarguments_addressed": true | false,
  "actionable_ask_present": true | false,
  "slides_rewritten": <count>,
  "weakest_dimension": "<which of the 5 scores was lowest across all slides>"
}

Rules:
- Be BRUTAL but constructive. A 3/5 is unacceptable for client-facing work.
- When rewriting, keep the same type and approximate length but make it SUBSTANTIALLY better.
- Check that the narrative arc flows — does each slide build on the previous?
- Verify counterarguments are addressed somewhere in the deck.
- Verify the last slide has a clear, actionable ask.
- Return ONLY valid JSON"""

    @staticmethod
    def _max_tokens_for_slides(num_slides: int) -> int:
        """Scale token budget dynamically: ~800 tokens per slide + overhead for scores."""
        return 10000 + (num_slides * 800)

    async def run(self, content: dict, brief: dict) -> dict:
        num_slides = len(content.get("slides", []))
        user_msg = f"""Completed Slides:
{json.dumps(content, indent=2)}

Original Research Brief (for fact-checking):
{json.dumps(brief, indent=2)}

Score every slide. Rewrite anything below 4/5 on any dimension."""

        result = await call_claude(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_msg,
            model="claude-sonnet-4-6",
            thinking_budget=5000,
            max_tokens=self._max_tokens_for_slides(num_slides),
        )

        review = parse_json_response(result["text"])
        review["_meta"] = {"agent": self.name, "elapsed": result["elapsed"]}
        return review

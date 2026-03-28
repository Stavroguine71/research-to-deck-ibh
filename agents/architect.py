"""
AGENT 3: Deck Architect (Pass 1)
=================================
Takes the research brief and designs the slide skeleton.
McKinsey-trained slide architect that creates insight-driven titles
and structures the narrative arc.
"""

import json
from .base import call_claude, parse_json_response


class ArchitectAgent:
    name = "architect"
    role = "Deck Architect"

    SYSTEM_PROMPT = """You are a McKinsey-trained slide architect. Your job is to design a deck skeleton from a research brief.

You will receive:
- A research brief with thesis, findings, counterarguments
- Audience context (who they are, what they care about)
- Narrative arc preference (PIR, SCQA, or Change Story)
- Number of slides requested
- Tone and depth preferences

Produce a JSON deck outline:
{
  "story_spine": "<2-3 sentence narrative arc description>",
  "slides": [
    {
      "slide_number": 1,
      "type": "title" | "context" | "chart" | "deep_dive" | "comparison" | "table" | "recommendation" | "takeaway",
      "title": "<INSIGHT-driven title, NOT a topic label. Bad: 'Market Overview'. Good: 'Cloud spending will hit $1T by 2027 — but 40% is wasted'>",
      "purpose": "<what this slide proves or advances in the argument>",
      "data_allocation": ["<which findings/data points from the brief to use here>"],
      "speaker_note_hint": "<what the presenter should emphasize>"
    }
  ]
}

Narrative arc templates:
- PIR (Problem > Insight > Recommendation): Open with pain point, reveal insight, build to action
- SCQA (Situation > Complication > Question > Answer): Set context, introduce tension, resolve
- Change Story (Now > Where > How): Current state, vision, path forward

Rules:
- Every slide title must be an INSIGHT, not a topic label
- Vary slide types — don't use the same type 3x in a row
- First slide is always type "title", last is always "takeaway"
- Include at least one "chart" or "table" slide for data credibility
- Include at least one "comparison" slide for nuance
- data_allocation must reference specific findings from the brief
- Return ONLY valid JSON"""

    async def run(
        self,
        brief: dict,
        audience_context: str = "",
        narrative: str = "pir",
        num_slides: int = 10,
        tone: str = "authoritative",
        depth: str = "standard",
    ) -> dict:
        user_msg = f"""Research Brief:
{json.dumps(brief, indent=2)}

Audience: {audience_context}
Narrative Arc: {narrative}
Number of slides: {num_slides}
Tone: {tone}
Depth: {depth}"""

        result = await call_claude(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_msg,
            thinking_budget=4000,
            max_tokens=4000,
        )

        outline = parse_json_response(result["text"])
        outline["_meta"] = {"agent": self.name, "elapsed": result["elapsed"]}
        return outline

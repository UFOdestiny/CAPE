"""Verbal-confidence elicitation: the ten-class prompt of LRMVerbal (Yoon et al., 2025, Listing 1) and its parser.

Used by the black-box run (``uqbench.blackbox.batch verbal``), where the verbal confidence is the only response
evidence the API exposes, and by the open-weight follow-up probe of the optional baselines.
"""
from __future__ import annotations

import re

VERBAL_CLASSES = {
    "almost no chance": 0.05, "highly unlikely": 0.15, "chances are slight": 0.25, "unlikely": 0.35,
    "less than even": 0.45, "better than even": 0.55, "likely": 0.65, "very good chance": 0.75,
    "highly likely": 0.85, "almost certain": 0.95,
}
VERBAL_PROMPT = (
    "Now, thoroughly assess your confidence in your answer above by evaluating your thinking process "
    "so far. Classify your confidence into one of the following classes based on how likely your "
    "answer is to be correct:\n"
    '- "Almost no chance" (0.0–0.1)\n- "Highly unlikely" (0.1–0.2)\n- "Chances are slight" (0.2–0.3)\n'
    '- "Unlikely" (0.3–0.4)\n- "Less than even" (0.4–0.5)\n- "Better than even" (0.5–0.6)\n'
    '- "Likely" (0.6–0.7)\n- "Very good chance" (0.7–0.8)\n- "Highly likely" (0.8–0.9)\n'
    '- "Almost certain" (0.9–1.0)\n'
    "Each category reflects the probability that your answer is correct. Format your confidence as "
    "**Confidence**: $CLASS where CLASS is one of the names (only the names without the probability "
    "ranges) of the classes above."
)
_CLASS_RE = re.compile("|".join(sorted(VERBAL_CLASSES, key=len, reverse=True)), re.IGNORECASE)


def parse_verbal(text: str) -> float | None:
    """Midpoint of the first verbal class mentioned (longest names matched first)."""
    m = _CLASS_RE.search(text)
    return VERBAL_CLASSES[m.group(0).lower()] if m else None

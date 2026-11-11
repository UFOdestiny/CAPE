"""Deterministic answer extraction, correctness labels, and answer-equivalence clustering.

Math answers are compared with ``math_verify`` and the numeric fallback below; multiple-choice answers by option
letter; open-domain QA answers by normalised alias match (TriviaQA convention). Labels of math answers these rules
reject can be revised by an LLM grader that sees the reference (``uqbench.labeling.regrade``); clusters never use
it, but they
use an LLM's reference-free judgment of whether two answers to the same question are equivalent.
A response without an extractable final answer is labelled incorrect and flagged ``no_answer`` so
that parse failures stay auditable.
"""
from __future__ import annotations

import re
from decimal import Decimal
from functools import lru_cache

THINK_END = "</think>"
HARMONY_FINAL = "<|channel|>final<|message|>"   # gpt-oss (harmony) start of the final answer


def split_thinking(text: str) -> tuple[str, str, bool]:
    """Split a generation into (reasoning, final response, closed).

    ``closed`` is False when the model never emitted ``</think>`` (typically a length cut-off);
    the whole text is then treated as reasoning and the final response is empty.
    """
    if THINK_END not in text:
        return text, "", False
    reasoning, response = text.rsplit(THINK_END, 1)
    return reasoning.replace("<think>", "", 1).strip(), response.strip(), True


def split_response(text: str, reasoning_format: str = "think") -> tuple[str, str, bool]:
    """(reasoning, final response, closed) for a backbone's output format.

    ``think``: reasoning inside ``<think>...</think>``; ``harmony``: gpt-oss analysis/final channels;
    ``none``: the whole output is the response (closedness is then decided by the finish reason,
    see ``uqbench.pipeline.generate``).
    """
    if reasoning_format == "none":
        return "", text.strip(), True
    if reasoning_format == "harmony":  # text decoded with special tokens kept
        if HARMONY_FINAL not in text:
            return text, "", False
        reasoning, response = text.rsplit(HARMONY_FINAL, 1)
        for tag in ("<|return|>", "<|end|>", "<|call|>"):
            response = response.replace(tag, "")
        return reasoning, response.strip(), True
    return split_thinking(text)


def extract_boxed(text: str) -> str | None:
    """Content of the last brace-balanced ``\\boxed{...}`` (or ``\\fbox{...}``)."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


_MC_PATTERNS = [
    re.compile(r"answer\s*(?:is|:)\s*\(?\**\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-J])\)?\s*$"),
]


def extract_choice(text: str) -> str | None:
    """Option letter from ``\\boxed{X}`` or an explicit "answer is X" statement."""
    boxed = extract_boxed(text)
    if boxed is not None:
        m = re.search(r"\b([A-J])\b", re.sub(r"\\text\{([^}]*)\}", r"\1", boxed))
        return m.group(1).upper() if m else None
    for pattern in _MC_PATTERNS:
        found = pattern.findall(text)
        if found:
            return found[-1].upper()
    return None


def normalize_qa(text: str) -> str:
    """TriviaQA / SQuAD normalisation: lowercase, drop punctuation, articles and extra spaces."""
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text).lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(response: str, task_type: str) -> str | None:
    if task_type == "mc":
        return extract_choice(response)
    boxed = extract_boxed(response)
    if task_type == "qa" and boxed is not None:
        return normalize_qa(boxed) or None
    return boxed


@lru_cache(maxsize=200_000)
def _parse_math(answer: str):
    from math_verify import parse

    return parse(f"\\boxed{{{answer}}}", parsing_timeout=5)


# Numeric spellings math_verify does not equate (mostly the physical answers of Minerva): "4.5e33" vs
# "4.5 \times 10^{33}", a leading "m \approx" or "\lambda =", and a unit in a text wrapper.
_SPACING = re.compile(r"\\[,;:! ]|~|\$|\\left|\\right")
_PREFIX = re.compile(r"^(?:\\?[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?\s*(?:=|\\approx|\\simeq)|\\approx)\s*")
_NUMBER = re.compile(r"^(-?)(\d+(?:\.\d*)?|\.\d+)(?:\s*(?:\\times|\\cdot)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?"
                     r"|\s*[eE]\s*([-+]?\d+))?")
_UNIT = re.compile(r"^(?:\s*(?:\\(?:text|mathrm|textrm|mbox)\s*\{[^{}0-9=]*\}|\^\s*\{?\\circ\}?|\\%))*\s*$")
SIG_FIGS = 3   # rounded answers count only when both sides carry at least this many significant figures
REL_TOL = Decimal("0.01")   # relative tolerance between approximate quantities (decimal point or exponent)


def _decimal(s: str):
    """(value, unit in the last place, significant figures, written as an integer) of a plain numeric answer,
    else None."""
    s = _PREFIX.sub("", _SPACING.sub(" ", s).strip()).strip()
    m = _NUMBER.match(s)
    if not m or not _UNIT.match(s[m.end():]):
        return None
    sign, digits, exp = m.group(1), m.group(2), m.group(3) or m.group(4) or "0"
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    sig = len(digits.replace(".", "").lstrip("0")) or 1
    integer = "." not in digits and m.group(3) is None and m.group(4) is None
    return Decimal(sign + digits) * Decimal(10) ** int(exp), Decimal(10) ** (int(exp) - decimals), sig, integer


def numeric_equal(gold: str, pred: str) -> bool:
    """Equal values in any of the spellings above; values within ``REL_TOL`` when both sides are written as
    approximate quantities (an integer must match exactly); or the same value rounded to the coarser of the two
    precisions when both have at least ``SIG_FIGS`` significant figures (20.4 vs 20.39)."""
    g, p = _decimal(gold), _decimal(pred)
    if g is None or p is None:
        return False
    if g[0] == p[0]:
        return True
    if not (g[3] or p[3]) and abs(g[0] - p[0]) <= REL_TOL * max(abs(g[0]), abs(p[0])):
        return True
    if min(g[2], p[2]) < SIG_FIGS:
        return False
    (fine, *_), (coarse, ulp, *_) = sorted((g, p), key=lambda x: x[1])
    return abs(fine - coarse) * 2 <= ulp


# A leading definition such as "z(t) =", "E_v(E_0) =" or "t_r =" in front of the only "=" of an answer.
_LHS = re.compile(r"^\s*\\?[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?(?:\([^()=]*\))?\s*=\s*(?=[^=]+$)")


def rhs(answer: str) -> str:
    return _LHS.sub("", answer, count=1)


def math_equal(gold: str, pred: str) -> bool:
    g2, p2 = rhs(gold), rhs(pred)
    if (g2, p2) != (gold, pred) and _math_equal(g2, p2):
        return True
    return _math_equal(gold, pred)


def _math_equal(gold: str, pred: str) -> bool:
    from math_verify import verify

    g, p = _parse_math(gold), _parse_math(pred)
    if not g or not p:
        return gold.strip() == pred.strip() or numeric_equal(gold, pred)
    try:
        return bool(verify(g, p, timeout_seconds=5)) or numeric_equal(gold, pred)
    except Exception:  # sympy edge cases; fall back to string identity
        return gold.strip() == pred.strip() or numeric_equal(gold, pred)


def is_correct(pred: str | None, gold, task_type: str) -> bool:
    """``gold`` is a string, or a list of accepted normalised aliases for ``qa``."""
    if pred is None:
        return False
    if task_type == "mc":
        return pred.upper() == gold.upper()
    if task_type == "qa":
        return normalize_qa(pred) in {normalize_qa(g) for g in gold}
    return math_equal(gold, pred)


def equivalent(a: str | None, b: str | None, task_type: str) -> bool:
    """Symmetric answer equivalence used to cluster sampled answers."""
    if a is None or b is None:
        return False
    if task_type == "mc":
        return a.upper() == b.upper()
    if task_type == "qa":
        return normalize_qa(a) == normalize_qa(b)
    return a.strip() == b.strip() or math_equal(a, b) or math_equal(b, a)


@lru_cache(maxsize=1)
def llm_equivalent() -> dict[str, set[frozenset[str]]]:
    """Math answer pairs an LLM judged equivalent without seeing the reference (``uqbench.labeling.regrade``),
    per question
    id; empty before that stage ran."""
    from uqbench.config import OUTPUT_DIR
    from uqbench.io import read_jsonl

    path = OUTPUT_DIR / "math_equiv" / "equivalent.jsonl"
    out: dict[str, set[frozenset[str]]] = {}
    for r in read_jsonl(path) if path.exists() else []:
        out.setdefault(r["uid"], set()).add(frozenset((r["a"], r["b"])))
    return out


def cluster_answers(answers: list[str | None], task_type: str, uid: str | None = None) -> list[int]:
    """Greedy equivalence clustering; each missing answer forms its own singleton cluster. With the question id, math
    answers the LLM judged equivalent (:func:`llm_equivalent`) also share a cluster."""
    same = llm_equivalent().get(uid, set()) if uid is not None and task_type == "math" else set()
    reps: list[str] = []
    labels: list[int] = []
    for ans in answers:
        for cid, rep in enumerate(reps):
            if rep is not None and ans is not None and (frozenset((rep, ans)) in same
                                                        or equivalent(rep, ans, task_type)):
                labels.append(cid)
                break
        else:
            reps.append(ans)
            labels.append(len(reps) - 1)
    return labels

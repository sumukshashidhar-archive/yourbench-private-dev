from __future__ import annotations
import re
import json
import random
import hashlib
from typing import Any, Mapping, Iterable
from dataclasses import dataclass

from loguru import logger


@dataclass(slots=True)
class _MCQ:
    choices: list[str]
    answer: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "_MCQ":
        return cls(
            choices=[str(c) for c in data.get("choices", [])],
            answer=str(data.get("answer", "")).strip().upper(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"choices": self.choices, "answer": self.answer}


def extract_content_from_xml_tags(full_content: str, xml_tag: str) -> str:
    """Return text inside ``<xml_tag>...</xml_tag>`` (best effort)."""
    try:
        closing = re.search(rf"<{xml_tag}>(.*?)</{xml_tag}>", full_content, re.DOTALL)
        if closing:
            return closing.group(1).strip()
        opening = re.search(rf"<{xml_tag}>(.*)", full_content, re.DOTALL)
        if opening:
            return opening.group(1).strip()
    except Exception as err:
        logger.error(f"extract_content_from_xml_tags failed for {xml_tag}: {err}")
    return ""


def parse_qa_pairs_from_response(raw_response: str) -> list[dict[str, Any]]:
    """
    Attempt to parse question-answer pairs from a raw LLM response.

    The function searches in this priority order:
        1. <output_json>...</output_json> tags.
        2. ```json fenced code blocks.
        3. Best-effort bracket-based extraction.

    If any candidate JSON is found, it attempts to parse it. If parsing
    succeeds and yields a list, it returns that list. Otherwise, it
    returns an empty list.

    Even if this returns an empty list, callers are expected to store
    the raw response (e.g., so the pipeline does not lose data).

    Args:
        raw_response (str): The complete raw response string from the model.

    Returns:
        A list of dict objects, each presumably containing
        question-answer information. If no valid parse is found,
        an empty list is returned.
    """
    if not isinstance(raw_response, str) or not raw_response.strip():
        return []

    candidates: list[str] = []
    tag_block = _extract_tag_content(raw_response, "output_json")
    if tag_block:
        candidates.append(_maybe_strip_triple_backticks(tag_block))
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.extend(_best_effort_json_extract(raw_response))

    return _parse_first_json_list(candidates)


def _extract_tag_content(text: str, tag: str) -> str:
    """Return inner text of ``<tag>`` if present."""
    try:
        match = re.search(rf"<{tag}\s*>([\s\S]*?)</{tag}>", text)
        return match.group(1).strip() if match else ""
    except Exception as err:
        logger.debug(f"_extract_tag_content error: {err}")
        return ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    """Remove surrounding ``` or ```json fences, if any."""
    if not isinstance(text_in, str):
        return ""
    try:
        match = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$", text_in)
        return match.group(1) if match else text_in
    except Exception as err:
        logger.debug(f"_maybe_strip_triple_backticks error: {err}")
        return text_in


def _best_effort_json_extract(full_text: str) -> list[str]:
    """Return bracket-delimited substrings that might be JSON."""
    if not isinstance(full_text, str):
        return []
    try:
        pattern = r"([\[{].*?[\]}])"
        return [m.strip() for m in re.findall(pattern, full_text, re.DOTALL)]
    except Exception as err:
        logger.debug(f"_best_effort_json_extract error: {err}")
        return []


def _attempt_json_parse(json_str: str) -> Any:
    """Return parsed JSON or ``None`` if invalid."""
    try:
        return json.loads(json_str)
    except Exception as err:
        logger.debug(f"JSON parse error: {err}")
        return None


def _parse_first_json_list(candidates: Iterable[str]) -> list[dict[str, Any]]:
    for cand in candidates:
        parsed = _attempt_json_parse(cand)
        if isinstance(parsed, list):
            return parsed
    return []


def shuffle_mcq(question_dict: dict) -> dict:
    """
    Shuffles MCQ choices randomly and ensures the correct answer is placed under a random label A-D.
    The final choices are labeled A., B., C., D. in order, but the correct answer may be under any of them.
    """
    mcq = _MCQ.from_mapping(question_dict)

    if not mcq.choices or not mcq.answer:
        return question_dict

    raw_choices = [c[3:].strip() for c in mcq.choices]
    try:
        answer_text = raw_choices[ord(mcq.answer) - ord("A")]
    except Exception:
        return question_dict

    seed = int(hashlib.sha256(repr((raw_choices, mcq.answer)).encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(raw_choices)

    new_letter = chr(ord("A") + raw_choices.index(answer_text))
    mcq.choices = [f"({chr(ord('A') + i)}) {t}" for i, t in enumerate(raw_choices)]
    mcq.answer = new_letter
    question_dict.update(mcq.to_dict())
    return question_dict

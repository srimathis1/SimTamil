"""Shared data helpers for the participant digital-twin survey scripts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = SCRIPT_DIR / "gss_bfi_questions.json"
DEFAULT_PARTICIPANTS_DIR = SCRIPT_DIR / "export" / "participants"
AGENT_DIRNAME = "agent_answers"
EVAL_DIRNAME = "agent_eval"
HUMAN_CONDITION = "human_wave1"
LIKERT_LABELS = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]


@dataclass(frozen=True)
class Item:
    key: str
    order: int
    kind: str
    qtype: str
    section: str
    trait: str | None
    reverse: bool
    question: str
    options: list[str]
    display_condition: dict[str, Any] | None


@dataclass
class Wave:
    raw: dict[str, Any]
    demographics: dict[str, Any]
    games: list[dict[str, Any]]
    by_key: dict[str, Any]

    def raw_answer(self, item: Item) -> Any:
        return self.by_key.get(item.key)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def find_participant_dir(username: str, participants_dir: Path) -> Path:
    direct = participants_dir / username
    if direct.is_dir():
        return direct
    matches = [p for p in participants_dir.iterdir() if p.is_dir() and p.name.lower() == username.lower()]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"Participant folder not found: {direct}")


def load_question_bank(path: Path) -> list[Item]:
    data = read_json(path)
    items: list[Item] = []
    for kind in ("gss", "bfi"):
        for entry in data.get(kind, []):
            key = entry["question_key"]
            section = str(entry.get("section") or "")
            items.append(Item(
                key=key,
                order=int(entry.get("order_index", len(items) + 1)),
                kind=kind,
                qtype="single_choice" if kind == "gss" else "likert5",
                section=section,
                trait=section if kind == "bfi" else None,
                reverse=key.endswith("_r"),
                question=str(entry.get("question", "")),
                options=list(entry.get("options") or LIKERT_LABELS),
                display_condition=entry.get("display_condition") or entry.get("condition"),
            ))
    return items


def load_wave(path: Path) -> Wave:
    raw = read_json(path)
    by_key: dict[str, Any] = {}
    for instrument in ("gss", "bfi"):
        for entry in raw.get(instrument, []) or []:
            key = entry.get("key") or entry.get("question_key")
            if key:
                ans = entry.get("answer")
                if isinstance(ans, dict):
                    ans = ans.get("value") or ans.get("label")
                by_key[key] = ans
    games = raw.get("behavioral_games", raw.get("games", [])) or []
    return Wave(
        raw=raw,
        demographics=raw.get("demographics") or {},
        games=games if isinstance(games, list) else [],
        by_key=by_key,
    )


def is_blank_answer(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def applicable(item: Item, demographics: dict[str, Any]) -> bool:
    condition = item.display_condition
    if not condition:
        return True
    field = condition.get("field")
    expected = condition.get("in", [])
    return demographics.get(field) in expected


def _answer_index(value: Any, options: list[str]) -> int | None:
    if isinstance(value, dict):
        value = value.get("value") or value.get("label")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if 1 <= value <= len(options) else None
    text = str(value).strip()
    match = re.match(r"^\s*(\d+)(?:\s*[-.:)]\s*|\s+|$)", text)
    if match:
        number = int(match.group(1))
        if 1 <= number <= len(options):
            return number
    for index, option in enumerate(options, 1):
        if text.casefold() == option.casefold():
            return index
    return None


def canon_answer(item: Item, value: Any) -> int | None:
    return _answer_index(value, item.options)


def mask_keys_for(item: Item, bank: list[Item]) -> set[str]:
    if item.kind == "bfi":
        return {other.key for other in bank if other.kind == "bfi" and other.section == item.section}
    return {item.key}


def value_text(item: Item, value: Any) -> str | None:
    index = canon_answer(item, value)
    if index is None:
        return None
    return item.options[index - 1]


def ordinal_distance(item: Item, first: int, second: int) -> int:
    return abs(first - second)

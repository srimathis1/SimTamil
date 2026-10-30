"""
Compare human Wave 1 and Wave 2 GSS/BFI answers.

Input:
    export/participants/<username>/all.json
    export/participants/<username>/wave2/wave2.json

Output:
    export/participants/<username>/wave2/human_wv_1_wv2.json

Usage:
    python compare_human_waves.py Mounees
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent
PARTICIPANTS_DIR = BASE_DIR / "export" / "participants"

# Standard five-point BFI response scale.
LIKERT_MIN = 1
LIKERT_MAX = 5

TRAITS = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
]


def load_json(path):
    """Load JSON and report a clear error if it cannot be read."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_question_key(item):
    """Get a stable question identifier."""
    for field in ("key", "question_key", "question_id", "id"):
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()

    # Fallback for files that do not contain question keys.
    for field in ("number", "order_index"):
        value = item.get(field)
        if value is not None:
            return f"number_{value}"

    question = item.get("question")
    if question:
        return str(question).strip().casefold()

    return None


def extract_raw_answer(item):
    """
    Handle both answer formats:
      Wave 1: "4 - Agree"
      Wave 2: {"value": "4", "label": "Agree"}
    """
    answer = item.get("answer")

    if isinstance(answer, dict):
        # Prefer the stored value; labels may be formatted differently.
        value = answer.get("value")
        if value is not None and str(value).strip():
            return value

        label = answer.get("label")
        if label is not None:
            return label

        return None

    return answer


def normalize_text(value):
    """Normalize text for fair categorical comparison."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    # Exclude explicit nonresponses from comparison.
    lowered = text.casefold()
    excluded = {
        "none",
        "null",
        "n/a",
        "na",
        "not answered",
        "not applicable",
        "not asked",
        "(not asked - not applicable to this participant)",
    }

    if lowered in excluded or lowered.startswith("(not asked"):
        return None

    return re.sub(r"\s+", " ", text).casefold()


def normalize_likert(value):
    """
    Convert BFI values such as:
      4
      "4"
      "4 - Agree"
      "Agree (4)"
    to an integer from 1 to 5.

    Returns None if a valid Likert score cannot be extracted.
    """
    if value is None:
        return None

    text = str(value).strip()

    # Typical stored values: "4", "4 - Agree".
    match = re.match(r"^\s*([1-5])(?:\s*(?:-|–|—|:)\s*.*)?$", text)
    if match:
        return int(match.group(1))

    # Alternative representation: "Agree (4)".
    match = re.search(r"\(([1-5])\)\s*$", text)
    if match:
        return int(match.group(1))

    return None


def answer_maps(items, answer_type):
    """Create a mapping from question key to normalized answer."""
    result = {}

    if not isinstance(items, list):
        return result

    for item in items:
        if not isinstance(item, dict):
            continue

        key = get_question_key(item)
        if key is None:
            continue

        raw_answer = extract_raw_answer(item)

        if answer_type == "bfi":
            normalized = normalize_likert(raw_answer)
        else:
            normalized = normalize_text(raw_answer)

        # Keep valid answered questions only.
        if normalized is not None:
            result[key] = {
                "answer": normalized,
                "question": item.get("question"),
                "section": item.get("section"),
            }

    return result


def compare_answers(wave1_items, wave2_items, answer_type):
    """Compare matched questions and return detailed results."""
    wave1 = answer_maps(wave1_items, answer_type)
    wave2 = answer_maps(wave2_items, answer_type)

    matched_keys = sorted(set(wave1) & set(wave2))
    wave1_only = sorted(set(wave1) - set(wave2))
    wave2_only = sorted(set(wave2) - set(wave1))

    details = []
    matching = 0
    different = 0

    for key in matched_keys:
        a1 = wave1[key]["answer"]
        a2 = wave2[key]["answer"]
        is_match = a1 == a2

        if is_match:
            matching += 1
        else:
            different += 1

        details.append({
            "question_key": key,
            "question": wave2[key]["question"] or wave1[key]["question"],
            "wave1_answer": a1,
            "wave2_answer": a2,
            "match": is_match,
        })

    compared = len(matched_keys)
    agreement = round(matching / compared * 100, 2) if compared else None

    return {
        "compared": compared,
        "matching": matching,
        "different": different,
        "agreement_percent": agreement,
        "wave1_answered_count": len(wave1),
        "wave2_answered_count": len(wave2),
        "unmatched_wave1_keys": wave1_only,
        "unmatched_wave2_keys": wave2_only,
        "details": details,
    }


def calculate_bfi_traits(wave1_items, wave2_items):
    """
    Calculate trait means using each question's section.

    Questions whose key ends in '_r' are reverse-coded:
        reverse score = 6 - original score

    This uses the custom questionnaire's section labels and keys,
    rather than assuming a different BFI-44 numbering scheme.
    """
    def collect(items):
        scores = {trait: [] for trait in TRAITS}
        item_scores = {}

        if not isinstance(items, list):
            return scores, item_scores

        for item in items:
            if not isinstance(item, dict):
                continue

            key = get_question_key(item)
            if not key:
                continue

            raw = extract_raw_answer(item)
            score = normalize_likert(raw)

            if score is None:
                continue

            section = str(item.get("section", "")).strip()
            trait = next(
                (t for t in TRAITS if t.casefold() == section.casefold()),
                None,
            )

            if trait is None:
                continue

            reverse_coded = key.casefold().endswith("_r")
            scored_value = LIKERT_MAX + LIKERT_MIN - score if reverse_coded else score

            scores[trait].append(scored_value)
            item_scores[key] = {
                "raw_score": score,
                "scored_value": scored_value,
                "reverse_coded": reverse_coded,
                "trait": trait,
            }

        return scores, item_scores

    w1_scores, w1_items = collect(wave1_items)
    w2_scores, w2_items = collect(wave2_items)

    trait_results = {}

    for trait in TRAITS:
        values1 = w1_scores[trait]
        values2 = w2_scores[trait]

        mean1 = round(sum(values1) / len(values1), 4) if values1 else None
        mean2 = round(sum(values2) / len(values2), 4) if values2 else None

        trait_results[trait.casefold()] = {
            "wave1_mean": mean1,
            "wave2_mean": mean2,
            "wave1_items_scored": len(values1),
            "wave2_items_scored": len(values2),
            "mean_difference_wave2_minus_wave1": (
                round(mean2 - mean1, 4)
                if mean1 is not None and mean2 is not None
                else None
            ),
        }

    # Item-level scoring details are useful for auditing reverse coding.
    return trait_results, w1_items, w2_items


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_human_waves.py <username>")
        sys.exit(1)

    username = sys.argv[1].strip()

    participant_dir = PARTICIPANTS_DIR / username
    wave1_path = participant_dir / "all.json"
    wave2_path = participant_dir / "wave2" / "wave2.json"
    output_path = participant_dir / "wave2" / "human_wv_1_wv2.json"

    wave1_data = load_json(wave1_path)
    wave2_data = load_json(wave2_path)

    # Compare GSS categorical answers.
    gss_result = compare_answers(
        wave1_data.get("gss", []),
        wave2_data.get("gss", []),
        "gss",
    )

    # Compare BFI Likert answers by their numeric values.
    bfi_result = compare_answers(
        wave1_data.get("bfi", []),
        wave2_data.get("bfi", []),
        "bfi",
    )

    trait_results, wave1_trait_items, wave2_trait_items = calculate_bfi_traits(
        wave1_data.get("bfi", []),
        wave2_data.get("bfi", []),
    )

    # Compare BFI trait means across waves.
    trait_comparison = {}
    for trait, values in trait_results.items():
        trait_comparison[trait] = values

    result = {
        "username": username,
        "participant_id": wave2_data.get(
            "participant_id", wave1_data.get("participant_id")
        ),
        "comparison": "Human Wave 1 vs Wave 2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "wave1": "all.json",
            "wave2": "wave2/wave2.json",
        },
        "methodology": {
            "matching": "Matched by question key",
            "gss_comparison": "Normalized case-insensitive categorical answers",
            "bfi_comparison": "Numeric Likert score comparison (1-5)",
            "bfi_reverse_coding": (
                "Question keys ending in _r are scored as 6 minus the raw score"
            ),
            "trait_means": "Mean of scored items grouped by section",
        },
        "gss": gss_result,
        "bfi": bfi_result,
        "bfi_trait_means": trait_comparison,
        "bfi_item_scoring": {
            "wave1": wave1_trait_items,
            "wave2": wave2_trait_items,
        },
    }

    # Save ONLY to the separate output file.
    # This script does not write to wave2.json.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    print(f"\nParticipant: {username}")
    print(
        f"GSS: compared={gss_result['compared']}, "
        f"matching={gss_result['matching']}, "
        f"different={gss_result['different']}, "
        f"agreement={gss_result['agreement_percent']}%"
    )
    print(
        f"BFI: compared={bfi_result['compared']}, "
        f"matching={bfi_result['matching']}, "
        f"different={bfi_result['different']}, "
        f"agreement={bfi_result['agreement_percent']}%"
    )

    print("\nBFI trait means (Wave 1 vs Wave 2):")
    for trait, values in trait_comparison.items():
        print(
            f"  {trait}: {values['wave1_mean']} vs {values['wave2_mean']}"
        )

    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()

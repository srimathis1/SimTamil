
"""
Export a participant's Wave 2 GSS and BFI answers from Supabase.

Usage:
    python extract_wave2.py prithika

Output:
    export/participants/<username>/wave2/wave2.json
"""

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def get_supabase_config():
    url = (
        os.getenv("SUPABASE_URL")
        or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        or ""
    ).rstrip("/")

    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        sys.exit(
            "Error: Add SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY to your .env file."
        )

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    return url, headers


def fetch_all(url, headers, table, **filters):
    """Fetch all rows from a Supabase table."""
    rows = []
    offset = 0

    while True:
        response = requests.get(
            f"{url}/rest/v1/{table}",
            headers=headers,
            params={
                "select": "*",
                "limit": 1000,
                "offset": offset,
                **filters,
            },
            timeout=60,
        )
        response.raise_for_status()

        batch = response.json()
        rows.extend(batch)

        if len(batch) < 1000:
            return rows

        offset += 1000


def format_answer(question, value):
    """Return the answer value and a readable option label if available."""
    options = question.get("options")

    if (
        isinstance(options, list)
        and str(value).isdigit()
        and 1 <= int(value) <= len(options)
    ):
        return {
            "value": value,
            "label": str(options[int(value) - 1]),
        }

    return {
        "value": value,
        "label": str(value),
    }


def main():
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("Usage: python extract_wave2.py <username>")

    username = sys.argv[1].strip()
    url, headers = get_supabase_config()

    print(f"Finding participant: {username}")

    participants = fetch_all(
        url,
        headers,
        "participants",
        username=f"eq.{username}",
    )

    if not participants:
        sys.exit(f"Participant not found: {username}")

    if len(participants) > 1:
        sys.exit(f"Multiple participants found: {username}")

    participant = participants[0]
    participant_id = participant["id"]

    # Fetch only Wave 2 answers for this participant.
    responses = fetch_all(
        url,
        headers,
        "wave2_responses",
        participant_id=f"eq.{participant_id}",
    )

    # Get question and instrument metadata.
    questions = fetch_all(url, headers, "questions")
    instruments = fetch_all(url, headers, "instruments")

    question_map = {
        question["id"]: question
        for question in questions
    }

    instrument_map = {
        instrument["id"]: instrument
        for instrument in instruments
    }

    result = {
        "wave": 2,
        "username": participant.get("username"),
        "participant_id": participant_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "summary": {},
        "gss": [],
        "bfi": [],
    }

    for response in responses:
        question = question_map.get(response["question_id"])

        if not question:
            print(
                f"Warning: Question {response['question_id']} "
                "was not found in questions."
            )
            continue

        instrument = instrument_map.get(
            question.get("instrument_id"), {}
        )
        slug = (instrument.get("slug") or "").lower()

        if slug not in ("gss", "bfi"):
            continue

        question_text = (
            question.get("prompt_english")
            or question.get("prompt_tamil")
            or ""
        )

        result[slug].append({
            "question_id": question["id"],
            "question_key": question.get("question_key"),
            "order_index": question.get("order_index"),
            "section": question.get("section"),
            "question": question_text,
            "answer": format_answer(
                question,
                response.get("answer_value"),
            ),
            "answered_at": response.get("answered_at"),
        })

    # Preserve questionnaire order.
    for section in ("gss", "bfi"):
        result[section].sort(
            key=lambda item: (
                item["order_index"] is None,
                item["order_index"]
                if item["order_index"] is not None
                else 0,
            )
        )

    result["summary"] = {
        "gss_answered": len(result["gss"]),
        "bfi_answered": len(result["bfi"]),
        "total_answered": (
            len(result["gss"]) + len(result["bfi"])
        ),
    }

    output_dir = (
        ROOT
        / "export"
        / "participants"
        / participant["username"]
        / "wave2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "wave2.json"
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nWave 2 export successful!")
    print(f"GSS answers: {len(result['gss'])}")
    print(f"BFI answers: {len(result['bfi'])}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as error:
        details = (
            error.response.text
            if error.response is not None
            else str(error)
        )
        sys.exit(f"Supabase API error: {details}")
    except requests.RequestException as error:
        sys.exit(f"Connection error: {error}")


"""
Export one participant, several participants, or all participants.

Usage:
    python3 scripts/export_participant.py raja_23
    python3 scripts/export_participant.py raja_23 meena_7
    python3 scripts/export_participant.py all
    python3 scripts/export_participant.py

Requirements:
    pip install requests python-dotenv

Environment variables in .env or your terminal:
    SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)
    SUPABASE_SERVICE_ROLE_KEY

Output:
    export/participants/<username>/
        all.json
        audio/
            <question_key>.webm

Only all.json and audio recordings are generated.
"""

import json
import os
import pathlib
import sys

import requests
from dotenv import load_dotenv


DEMO_IDS = [
    "age_group",
    "gender",
    "marital_status",
    "education",
    "children",
    "religion",
    "work_status",
    "residence",
]


def applicable(q, demographics):
    """Check whether a question applies to the participant."""
    condition = q.get("condition")

    if not condition:
        return True

    return (
        demographics or {}
    ).get(condition["field"]) in condition.get("in", [])


def answer_label(q, value):
    """Convert an answer into a readable label."""
    options = q.get("options")

    if (
        q.get("type") == "likert5"
        and isinstance(options, list)
        and str(value).isdigit()
        and 1 <= int(value) <= len(options)
    ):
        return f"{value} - {options[int(value) - 1]}"

    if q.get("type") == "slider" and isinstance(options, dict):
        return f"{options.get('unit', '')}{value}"

    return str(value)


def build_reports(
    participant,
    instruments,
    questions,
    responses,
    recordings,
    demo_text=None,
):
    """
    Build structured participant data.

    Returns:
        A dictionary containing all participant data.

    No text report files are generated.
    """
    demographics = participant.get("demographics") or {}

    answers = {
        r["question_id"]: r["answer_value"]
        for r in responses
    }

    recording_map = {
        r["question_id"]: r
        for r in recordings
    }

    data = {
        "username": participant["username"],
        "participant_id": participant["id"],
        "demographics": demographics,
    }

    # Add readable demographic question labels.
    demographic_questions = {}

    for i, key in enumerate(DEMO_IDS):
        demographic_questions[key] = {
            "question": (demo_text or {}).get(key, key),
            "answer": demographics.get(key),
        }

    data["demographics_questions"] = demographic_questions

    # Add interview, GSS, BFI, and behavioral-game data.
    for instrument in sorted(
        instruments,
        key=lambda item: item["order_index"],
    ):
        slug = instrument["slug"]

        if slug not in {
            "interview",
            "gss",
            "bfi",
            "behavioral_games",
        }:
            continue

        instrument_questions = sorted(
            (
                q
                for q in questions
                if q["instrument_id"] == instrument["id"]
                and q["is_active"]
            ),
            key=lambda q: q["order_index"],
        )

        items = []

        for q in instrument_questions:
            question_text = (
                q.get("prompt_tamil")
                or q.get("prompt_english")
                or ""
            )

            if not applicable(q, demographics):
                answer = (
                    "Not asked - not applicable to this participant"
                )

            elif slug == "interview":
                recording = recording_map.get(q["id"])

                if not recording:
                    answer = "(not answered)"

                elif recording.get("transcript"):
                    answer = recording["transcript"]

                else:
                    answer = {
                        "audio_file": (
                            f"audio/{q['question_key']}.webm"
                        ),
                        "duration_seconds": recording.get(
                            "duration_seconds"
                        ),
                        "transcript": None,
                    }

            else:
                if q["id"] in answers:
                    answer = answer_label(q, answers[q["id"]])
                else:
                    answer = "(not answered)"

            items.append(
                {
                    "number": q["order_index"],
                    "key": q["question_key"],
                    "section": q.get("section"),
                    "question": question_text,
                    "answer": answer,
                }
            )

        data[slug] = items

    return data


def main():
    # Load environment variables from scripts/.env.
    load_dotenv(
        pathlib.Path(__file__).with_name(".env")
    )

    url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        or ""
    ).rstrip("/")

    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        sys.exit(
            "Set SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) "
            "and SUPABASE_SERVICE_ROLE_KEY in .env "
            "or the environment."
        )

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

    def fetch(table, **filters):
        """Fetch all rows from a Supabase REST table."""
        rows = []
        offset = 0

        while True:
            response = requests.get(
                f"{url}/rest/v1/{table}",
                headers=headers,
                timeout=60,
                params={
                    "select": "*",
                    "limit": 1000,
                    "offset": offset,
                    "order": "id",
                    **filters,
                },
            )

            response.raise_for_status()
            batch = response.json()
            rows.extend(batch)

            if len(batch) < 1000:
                return rows

            offset += 1000

    # Read participant usernames or IDs from command-line arguments.
    names = sys.argv[1:]

    if not names:
        entered = input(
            "Participant username or ID "
            "(comma separated, or 'all'): "
        )

        names = [
            name.strip()
            for name in entered.split(",")
            if name.strip()
        ]

    if not names:
        sys.exit("No participant given.")

    if [name.lower() for name in names] == ["all"]:
        people = fetch("participants")
    else:
        people = []
        seen_ids = set()

        for name in names:
            found = fetch(
                "participants",
                username=f"eq.{name}",
            )

            # If no username matched, try participant ID.
            if not found:
                found = fetch(
                    "participants",
                    id=f"eq.{name}",
                )

            if not found:
                print(f"  not found: {name}")
                continue

            for person in found:
                if person["id"] not in seen_ids:
                    people.append(person)
                    seen_ids.add(person["id"])

    if not people:
        sys.exit("Nothing to export.")

    # Fetch instrument definitions and questions once.
    instruments = fetch("instruments")
    questions = fetch("questions")

    # Map question IDs to their question keys for audio filenames.
    question_keys = {
        q["id"]: q["question_key"]
        for q in questions
    }

    # Load demographic question labels, if questions.json exists.
    demo_text = {}

    questions_file = pathlib.Path(__file__).with_name(
        "questions.json"
    )

    if questions_file.exists():
        question_config = json.loads(
            questions_file.read_text(encoding="utf-8")
        )

        demo_text = {
            key: text
            for key, (_, text, _) in zip(
                DEMO_IDS,
                question_config.get("demographics", []),
            )
        }

    # Export each selected participant.
    for participant in people:
        participant_id = participant["id"]
        username = participant["username"]

        responses = fetch(
            "responses",
            participant_id=f"eq.{participant_id}",
        )

        recordings = fetch(
            "interview_recordings",
            participant_id=f"eq.{participant_id}",
        )

        # Build the structured JSON data.
        data = build_reports(
            participant,
            instruments,
            questions,
            responses,
            recordings,
            demo_text,
        )

        output_dir = (
            pathlib.Path("export")
            / "participants"
            / username
        )

        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Save only all.json.
        json_path = output_dir / "all.json"

        json_path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Download audio recordings into audio/.
        downloaded = 0

        for recording in recordings:
            question_id = recording["question_id"]
            question_key = question_keys.get(
                question_id,
                str(question_id),
            )

            destination = (
                audio_dir / f"{question_key}.webm"
            )

            # Skip files that already exist.
            if destination.exists():
                downloaded += 1
                continue

            storage_path = recording["storage_path"]

            audio_url = (
                f"{url}/storage/v1/object/"
                f"interview-audio/{storage_path}"
            )

            try:
                audio_response = requests.get(
                    audio_url,
                    headers=headers,
                    timeout=120,
                )

                if audio_response.status_code != 200:
                    print(
                        "  could not download",
                        storage_path,
                        audio_response.status_code,
                    )
                    continue

                destination.write_bytes(
                    audio_response.content
                )

                downloaded += 1

            except requests.RequestException as error:
                print(
                    f"  audio download failed for "
                    f"{storage_path}: {error}"
                )

        print(
            f"{username}: {len(responses)} answers, "
            f"{downloaded}/{len(recordings)} audio files "
            f"-> {output_dir}"
        )


if __name__ == "__main__":
    main()

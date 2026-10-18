
import os
import re
import csv
import json
import time
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from google import genai

# ============================================================
# SIMTAMIL: EXISTING REFLECTION EVALUATION
#
# Method:
#   1. Read existing expert reflections
#   2. Retrieve candidate evidence from Wave 1 only
#   3. Ask Gemini to propose claim-level classifications
#   4. Validate returned evidence excerpts
#   5. Calculate descriptive statistics using Python
#
# Features:
#   - Small batches to reduce prompt size
#   - 15-second request spacing
#   - Stop on quota errors
#   - Save after every batch
#   - Resume completed claims on the next run
#   - Never regenerate reflections
#   - Never load Wave 2
# ============================================================

# ---------------- CONFIGURATION ----------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not API_KEY:
    raise RuntimeError(
        "API key missing. Add GEMINI_API_KEY to your project-root .env file."
    )

client = genai.Client(api_key=API_KEY)

ROOT = Path(__file__).resolve().parents[2]

PARTICIPANT_DIR = ROOT / "data" / "participant01"
REFLECTION_DIR = ROOT / "data" / "Expert_reflection"
TRANSCRIPT_DIR = PARTICIPANT_DIR / "Transcripts"
SURVEY_FILE = PARTICIPANT_DIR / "surveys_wave1.txt"

OUTPUT_DIR = REFLECTION_DIR / "evaluation_automated"
CLAIMS_CSV = OUTPUT_DIR / "claim_evidence_audit.csv"
METRICS_CSV = OUTPUT_DIR / "reflection_metrics.csv"
REPORT_TXT = OUTPUT_DIR / "evaluation_report.txt"

# Small requests are safer for free-tier token limits.
BATCH_SIZE = 3
REQUEST_DELAY_SECONDS = 15
MAX_CHUNK_CHARS = 600
CHUNK_OVERLAP = 100
EVIDENCE_CHUNKS_PER_CLAIM = 1

EXPERTS = {
    "PSYCHOLOGIST": "psychologist_reflection.txt",
    "BEHAVIORAL_ECONOMIST": "behavioral_economist_reflection.txt",
    "POLITICAL_SCIENTIST": "political_scientist_reflection.txt",
    "DEMOGRAPHER": "demographer_reflection.txt",
}

STATUSES = {
    "supported",
    "contradicted",
    "insufficient_evidence",
    "interpretation",
    "not_assessed",
}

FIELDS = [
    "expert",
    "reflection_file",
    "claim_id",
    "claim_text",
    "status",
    "evidence_source",
    "evidence_chunk_id",
    "evidence_excerpt",
    "reason",
    "human_review_needed",
    "human_verified",
]


# ---------------- FILE HELPERS ----------------

def read_text(path):
    return path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )


def load_sources():
    if not SURVEY_FILE.exists():
        raise FileNotFoundError(
            f"Wave 1 survey not found: {SURVEY_FILE}"
        )

    if not TRANSCRIPT_DIR.exists():
        raise FileNotFoundError(
            f"Transcript directory not found: {TRANSCRIPT_DIR}"
        )

    sources = [{
        "source": str(SURVEY_FILE.relative_to(ROOT)),
        "text": read_text(SURVEY_FILE),
    }]

    transcript_files = sorted(
        path for path in TRANSCRIPT_DIR.rglob("*.txt")
        if path.is_file()
    )

    for path in transcript_files:
        sources.append({
            "source": str(path.relative_to(ROOT)),
            "text": read_text(path),
        })

    print(f"Evidence files loaded: {len(sources)}")
    print("Wave 2 loaded: NO")

    if len(sources) == 1:
        print("WARNING: No interview transcript files found.")

    return sources


# ---------------- EVIDENCE RETRIEVAL ----------------

def split_chunks(text):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def tokenize(text):
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def retrieve_evidence(claim, sources):
    """
    Select one candidate evidence chunk by lexical overlap.
    This is retrieval, not proof that the claim is accurate.
    """
    claim_tokens = tokenize(claim)
    best = None
    best_score = -1.0

    for source in sources:
        for chunk_number, chunk in enumerate(
            split_chunks(source["text"]), start=1
        ):
            chunk_tokens = tokenize(chunk)

            score = (
                len(claim_tokens & chunk_tokens) / len(claim_tokens)
                if claim_tokens else 0.0
            )

            if score > best_score:
                best_score = score
                best = {
                    "source": source["source"],
                    "chunk_id": chunk_number,
                    "text": chunk,
                    "score": score,
                }

    # Do not attach arbitrary evidence when there is no lexical match.
    if best is None or best_score <= 0:
        return []

    return [best]


# ---------------- CLAIM EXTRACTION ----------------

def split_claims(text):
    """
    Sentence-level candidate extraction.
    Manually review sentences that contain multiple claims.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(r"\n+|(?<=[.!?])\s+", text)

    return [
        re.sub(r"\s+", " ", part).strip()
        for part in parts
        if len(part.strip()) >= 25
    ]


def make_claim_payload(expert, filename, claims, sources):
    payload = []

    for number, claim in enumerate(claims, start=1):
        claim_id = f"{expert}_{number:04d}"
        evidence = retrieve_evidence(claim, sources)

        payload.append({
            "expert": expert,
            "reflection_file": filename,
            "claim_id": claim_id,
            "claim_text": claim,
            "candidate_evidence": evidence,
        })

    return payload


# ---------------- JSON HANDLING ----------------

def parse_json_response(text):
    text = text.strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise ValueError("Response did not contain a JSON object.")

    return json.loads(text[start:end + 1])


def is_quota_error(exc):
    message = str(exc).lower()

    return any(term in message for term in (
        "429",
        "resource_exhausted",
        "quota exceeded",
        "quota_exceeded",
        "rate limit",
    ))


# ---------------- LLM CLAIM EVALUATION ----------------

def build_prompt(expert, batch):
    instructions = {
        "PSYCHOLOGIST": (
            "Focus on personality, emotions, stated motivations, "
            "coping, and decision-making. Do not infer a clinical diagnosis."
        ),
        "BEHAVIORAL_ECONOMIST": (
            "Focus on economic choices, risk, time preferences, "
            "trade-offs, and game responses."
        ),
        "POLITICAL_SCIENTIST": (
            "Focus on institutional trust, authority, civic attitudes, "
            "social norms, and political or public preferences."
        ),
        "DEMOGRAPHER": (
            "Focus on demographic facts, family structure, education, "
            "living context, employment, and socioeconomic indicators."
        ),
    }.get(expert, "Focus on the expert's relevant domain.")

    compact = []

    for item in batch:
        evidence = item["candidate_evidence"]

        compact.append({
            "claim_id": item["claim_id"],
            "claim": item["claim_text"],
            "evidence": [
                {
                    "source": ev["source"],
                    "chunk_id": ev["chunk_id"],
                    "text": ev["text"],
                }
                for ev in evidence
            ],
        })

    return f"""
You are an evidence auditor for the SimTamil research project.

EXPERT ROLE:
{expert}

DOMAIN GUIDANCE:
{instructions}

Evaluate the reflection claims ONLY against the supplied Wave 1
survey and interview evidence.

Do not use Wave 2.
Do not regenerate or rewrite reflections.
Do not reward fluent or professional writing.
Do not assume a claim is true because it sounds plausible.
Do not invent source quotations, survey answers, or source references.

Use exactly one status per claim:

supported:
The evidence directly supports the factual claim.

contradicted:
The evidence explicitly conflicts with the factual claim.

insufficient_evidence:
The available evidence does not establish the factual claim.
This does not mean the claim is false.

interpretation:
The claim is primarily an inference, theoretical explanation,
or interpretation rather than a directly verifiable fact.
Explain whether the evidence gives a reasonable basis for it.

If a factual claim has no relevant evidence, choose insufficient_evidence.
If a source excerpt is not provided, do not invent one.
An interpretation must not be treated as a verified fact.

Return valid JSON only:

{{
  "results": [
    {{
      "claim_id": "exact claim ID",
      "status": "supported",
      "evidence_source": "exact source path or empty string",
      "evidence_chunk_id": 1,
      "evidence_excerpt": "short exact excerpt or empty string",
      "reason": "brief evidence-based explanation"
    }}
  ]
}}

Return each supplied claim ID exactly once.

CLAIMS:
{json.dumps(compact, ensure_ascii=False)}
"""


def make_unassessed(item, reason):
    return {
        "expert": item["expert"],
        "reflection_file": item["reflection_file"],
        "claim_id": item["claim_id"],
        "claim_text": item["claim_text"],
        "status": "not_assessed",
        "evidence_source": "",
        "evidence_chunk_id": "",
        "evidence_excerpt": "",
        "reason": reason,
        "human_review_needed": "yes",
        "human_verified": "no",
    }


def evaluate_batch(batch):
    expert = batch[0]["expert"]
    prompt = build_prompt(expert, batch)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

        parsed = parse_json_response(response.text)
        returned = parsed.get("results", [])

    except Exception as exc:
        print(f"API request failed: {exc}")

        if is_quota_error(exc):
            raise RuntimeError(
                "GEMINI_QUOTA_EXHAUSTED"
            ) from exc

        return [
            make_unassessed(item, f"API or parsing error: {exc}")
            for item in batch
        ]

    result_map = {}

    for result in returned:
        claim_id = str(result.get("claim_id", "")).strip()

        if claim_id in {item["claim_id"] for item in batch}:
            if claim_id not in result_map:
                result_map[claim_id] = result

    final_rows = []

    for item in batch:
        claim_id = item["claim_id"]
        result = result_map.get(claim_id)

        if result is None:
            final_rows.append(
                make_unassessed(item, "Missing result from model.")
            )
            continue

        status = str(result.get("status", "")).strip().lower()

        if status not in STATUSES:
            status = "not_assessed"

        source = str(result.get("evidence_source", "")).strip()
        excerpt = str(result.get("evidence_excerpt", "")).strip()
        chunk_id = result.get("evidence_chunk_id", "")

        # Verify that the returned source/excerpt exists in the
        # evidence chunk that was actually supplied for this claim.
        valid_reference = any(
            ev["source"] == source
            and str(ev["chunk_id"]) == str(chunk_id)
            and excerpt
            and excerpt in ev["text"]
            for ev in item["candidate_evidence"]
        )

        if not valid_reference:
            source = ""
            excerpt = ""
            chunk_id = ""

            if status in {"supported", "contradicted"}:
                status = "insufficient_evidence"

        final_rows.append({
            "expert": item["expert"],
            "reflection_file": item["reflection_file"],
            "claim_id": claim_id,
            "claim_text": item["claim_text"],
            "status": status,
            "evidence_source": source,
            "evidence_chunk_id": chunk_id,
            "evidence_excerpt": excerpt,
            "reason": str(result.get("reason", "")),
            "human_review_needed": "yes",
            "human_verified": "no",
        })

    return final_rows


# ---------------- RESUME SUPPORT ----------------

def load_saved_rows():
    """
    Reuse completed rows from a previous run.
    not_assessed rows are retried on a later run.
    """
    if not CLAIMS_CSV.exists():
        return {}

    saved = {}

    try:
        with CLAIMS_CSV.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            for row in csv.DictReader(file):
                claim_id = row.get("claim_id", "")
                status = row.get("status", "")

                if not claim_id:
                    continue

                # Preserve completed rows, including manual audit
                # decisions. Only retry unassessed claims.
                if status != "not_assessed":
                    saved[claim_id] = row

    except Exception as exc:
        print(f"Could not load previous progress: {exc}")

    return saved


def save_claim_rows(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Always persist a complete snapshot.
    with CLAIMS_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------- STATISTICAL METRICS ----------------

def percentage(numerator, denominator):
    if denominator == 0:
        return None
    return round(100 * numerator / denominator, 2)


def calculate_metrics(rows):
    metrics = []

    for expert in EXPERTS:
        expert_rows = [
            row for row in rows
            if row["expert"] == expert
        ]

        counts = Counter(row["status"] for row in expert_rows)

        supported = counts["supported"]
        contradicted = counts["contradicted"]
        insufficient = counts["insufficient_evidence"]

        denominator = supported + contradicted + insufficient

        traceable = sum(
            1 for row in expert_rows
            if row["status"] in {
                "supported",
                "contradicted",
                "insufficient_evidence",
            }
            and row.get("evidence_source", "").strip()
            and row.get("evidence_excerpt", "").strip()
        )

        verified = [
            row for row in expert_rows
            if row.get("human_verified", "").strip().lower() == "yes"
            and row["status"] in {
                "supported",
                "contradicted",
                "insufficient_evidence",
            }
        ]

        human_counts = Counter(row["status"] for row in verified)
        human_denominator = sum(
            human_counts[s]
            for s in (
                "supported",
                "contradicted",
                "insufficient_evidence",
            )
        )

        metrics.append({
            "expert": expert,
            "candidate_claims": len(expert_rows),
            "supported_llm_proposed": supported,
            "contradicted_llm_proposed": contradicted,
            "insufficient_llm_proposed": insufficient,
            "interpretations_llm_proposed": counts["interpretation"],
            "not_assessed": counts["not_assessed"],
            "verifiable_claims_assessed": denominator,
            "llm_support_rate_pct": percentage(supported, denominator),
            "llm_contradiction_rate_pct": percentage(
                contradicted, denominator
            ),
            "llm_insufficient_rate_pct": percentage(
                insufficient, denominator
            ),
            "llm_traceability_pct": percentage(traceable, denominator),
            "human_verified_claims": len(verified),
            "human_support_rate_pct": percentage(
                human_counts["supported"], human_denominator
            ),
            "human_contradiction_rate_pct": percentage(
                human_counts["contradicted"], human_denominator
            ),
        })

    return metrics


def save_metrics(metrics):
    if not metrics:
        return

    with METRICS_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metrics[0].keys())
        )
        writer.writeheader()
        writer.writerows(metrics)


def save_report(metrics):
    def fmt(value):
        return "N/A" if value is None else f"{value:.2f}%"

    lines = [
        "=" * 72,
        "SIMTAMIL - EXISTING REFLECTION EVALUATION",
        "=" * 72,
        f"Model: {MODEL_NAME}",
        "Source: Wave 1 survey and interview transcripts",
        "Wave 2 loaded: NO",
        "Reflection files regenerated: NO",
        "",
        "METHOD",
        "Claim classifications are proposed by an LLM.",
        "Python calculates descriptive metrics from those labels.",
        "Human review is required before claiming factual accuracy.",
        "Lexical retrieval is not proof of factual support.",
        "",
        "FORMULAS",
        "Support = supported / (supported + contradicted + insufficient) * 100",
        "Contradiction = contradicted / same denominator * 100",
        "Insufficient evidence = insufficient / same denominator * 100",
        "Traceability = claims with source and excerpt / denominator * 100",
        "",
    ]

    for row in metrics:
        lines.extend([
            "-" * 72,
            f"EXPERT: {row['expert']}",
            f"Candidate claims: {row['candidate_claims']}",
            f"Assessed factual claims: {row['verifiable_claims_assessed']}",
            f"Supported (LLM proposed): {row['supported_llm_proposed']}",
            f"Contradicted (LLM proposed): {row['contradicted_llm_proposed']}",
            f"Insufficient evidence: {row['insufficient_llm_proposed']}",
            f"Interpretations: {row['interpretations_llm_proposed']}",
            f"Not assessed: {row['not_assessed']}",
            f"LLM support rate: {fmt(row['llm_support_rate_pct'])}",
            f"LLM contradiction rate: {fmt(row['llm_contradiction_rate_pct'])}",
            f"LLM insufficient rate: {fmt(row['llm_insufficient_rate_pct'])}",
            f"LLM traceability: {fmt(row['llm_traceability_pct'])}",
            f"Human-verified claims: {row['human_verified_claims']}",
            f"Human support rate: {fmt(row['human_support_rate_pct'])}",
            "",
        ])

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


# ---------------- MAIN ----------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIMTAMIL EXISTING REFLECTION EVALUATION")
    print("=" * 60)

    sources = load_sources()
    saved = load_saved_rows()

    all_rows_by_id = dict(saved)
    quota_stopped = False

    for expert, filename in EXPERTS.items():
        reflection_path = REFLECTION_DIR / filename

        if not reflection_path.exists():
            print(f"Missing reflection: {reflection_path}")
            continue

        claims = split_claims(read_text(reflection_path))
        payload = make_claim_payload(
            expert, filename, claims, sources
        )

        pending = [
            item for item in payload
            if item["claim_id"] not in all_rows_by_id
        ]

        print(
            f"\n{expert}: {len(claims)} claims; "
            f"{len(pending)} remaining"
        )

        if not pending:
            print("All claims already have saved results.")
            continue

        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]

            print(
                f"Batch {start // BATCH_SIZE + 1}: "
                f"{len(batch)} claims"
            )

            try:
                batch_rows = evaluate_batch(batch)

            except RuntimeError as exc:
                if "GEMINI_QUOTA_EXHAUSTED" in str(exc):
                    print(
                        "\nQuota exhausted. Saving progress and stopping."
                    )
                    quota_stopped = True
                    break
                raise

            for row in batch_rows:
                all_rows_by_id[row["claim_id"]] = row

            # Save after EVERY batch.
            save_claim_rows(list(all_rows_by_id.values()))

            print("Progress saved.")

            if start + BATCH_SIZE < len(pending):
                print(
                    f"Waiting {REQUEST_DELAY_SECONDS} seconds..."
                )
                time.sleep(REQUEST_DELAY_SECONDS)

        if quota_stopped:
            break

        # Space requests between experts as well.
        time.sleep(REQUEST_DELAY_SECONDS)

    all_rows = list(all_rows_by_id.values())

    if all_rows:
        metrics = calculate_metrics(all_rows)
        save_metrics(metrics)
        save_report(metrics)

    print("\n" + "=" * 60)
    print("EVALUATION STOPPED" if quota_stopped else "EVALUATION FINISHED")
    print("=" * 60)
    print(f"Claim audit: {CLAIMS_CSV}")
    print(f"Metrics:     {METRICS_CSV}")
    print(f"Report:      {REPORT_TXT}")

    if quota_stopped:
        print(
            "\nRun the same command again after the quota resets. "
            "Completed claims will be reused."
        )

    print(
        "\nLLM-proposed statistics are preliminary. "
        "Human verification is required for final research results."
    )


if __name__ == "__main__":
    main()

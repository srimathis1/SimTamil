#!/usr/bin/env python3
"""
expert_sur_int.py -- build the 5 expert reflections for ONE participant from
BOTH the structured survey data (all.json) and the interview transcript
(interview_transcript_ta.txt). Observations are written in TAMIL by default.

Run from the Data_Processor folder (next to extract.py):

    python expert_sur_int.py Mouneeswaran

Reads : export/participants/<username>/all.json
        export/participants/<username>/interview_transcript_ta.txt
Writes: export/participants/<username>/expert_ta/        (Tamil, default)
            demographer.txt
            psychologist.txt
            behavioral_economist.txt
            political_scientist.txt
            sociologist.txt
            _run_log.json          (model, flags, input sizes, observation counts,
                                    Tamil-script ratio per expert)

Data source (--source):
    both       survey + interview (default)
    interview  transcript only    (same as the old --no-survey)
    survey     all.json only

Output folders (so different conditions never overwrite each other):
    both      -> expert_ta/                  (English: expert/)
    interview -> expert_interview_only_ta/   (English: expert_interview_only/)
    survey    -> expert_survey_only_ta/      (English: expert_survey_only/)

Useful flags:
    --lang ta|en         language of the observations (default: ta)
    --dry-run            build and save the exact prompts, make NO API calls
    --force              regenerate files that already exist (default: skip them,
                         so a re-run never spends API budget on finished experts)
    --only psychologist sociologist     run just some experts
    --exclude TRUST FAIR gss            drop survey items/modules BEFORE the model
                                        sees them (use this for questions you plan
                                        to hold out and predict later)
    --source both|interview|survey   which data the experts see (default: both)
    --require-evidence   ask for a short supporting quote after every observation
    --model, --temperature, --participants-dir, --survey-file, --out-dir-name

API key: read from the GEMINI_API_KEY environment variable or from a .env file in
this folder. It is never printed, logged, or written anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

# Windows consoles default to cp1252 and crash on Tamil text; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ( "gemini-2.5-flash-lite","gemini-flash-latest")
SCRIPT_DIR = Path(__file__).resolve().parent
PLACEHOLDER = "<PARTICIPANT_DATA_BLOCK>"
MIN_TAMIL_RATIO = 0.6  # warn if less than this share of letters is Tamil script

# --------------------------------------------------------------------------
# The five expert prompts (exactly as designed; only the data block is filled)
# --------------------------------------------------------------------------
EXPERT_PROMPTS: dict[str, str] = {}

EXPERT_PROMPTS["demographer"] = """You are an expert Demographer (with a PhD) specializing in population dynamics, socio-economic status, and household structures in South Asia and India.

TASK:
Analyze the provided participant self-report data, which contains BOTH structured survey responses (Section 1) AND an open-ended qualitative interview transcript (Section 2). Cross-reference these sources to generate a structured list of domain-specific observations and inferences regarding their demographic background, socio-economic standing, and life stage.

KEY FOCUS AREAS:
1. Exact or estimated age, life stage, and generation.
2. Gender identity and familial role (e.g., primary earner, caregiver, dependent).
3. Geographical setting (District/City in Tamil Nadu, Urban vs. Semi-Urban vs. Rural classification, migration history).
4. Educational attainment and language/literacy profile.
5. Employment stability, occupation, personal/household income bracket, and economic mobility.
6. Household composition (nuclear vs. multi-generational joint family, number of dependents).

SYNTHESIS INSTRUCTIONS:
- Use Section 1 survey data for exact quantitative figures (income bracket, formal education level).
- Use Section 2 interview transcripts to understand the lived reality behind those numbers (e.g., income volatility, informal side jobs, dependents living elsewhere).

OUTPUT FORMAT:
- Generate between 5 and 20 concise, factual bullet-point observations.
- Base observations directly on the provided data or make well-reasoned domain inferences.
- Refer to the participant as "The interviewee" or "The participant".

PARTICIPANT DATA:
<PARTICIPANT_DATA_BLOCK>"""

EXPERT_PROMPTS["psychologist"] = """You are an expert Psychologist (with a PhD) specializing in adult personality, emotional resilience, stress coping mechanisms, and personal autonomy.

TASK:
Analyze the provided participant self-report data, which contains BOTH structured survey responses (Section 1) AND an open-ended qualitative interview transcript (Section 2). Cross-reference their quantitative personality scores (BFI-44) with their qualitative life stories to generate a structured list of observations regarding their psychological profile.

KEY FOCUS AREAS:
1. Core personality traits (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness) validated across survey scores and real-world story examples.
2. Degree of personal autonomy versus emotional interdependence with family/relatives.
3. Psychological coping strategies when facing life stress, illness, or financial pressure.
4. Locus of control (internal agency vs. external fatalism/destiny).
5. Emotional resilience, self-esteem, and subjective life satisfaction.

SYNTHESIS INSTRUCTIONS:
- Use Section 1 BFI-44 scores as quantitative baseline anchors.
- Use Section 2 interview narratives to explain *why* they exhibit those scores (e.g., linking a high Neuroticism score on the survey to specific past family trauma or health anxiety mentioned in the interview).

OUTPUT FORMAT:
- Generate between 5 and 20 concise bullet-point psychological observations and inferences.
- Focus on latent psychological mechanisms that drive decision-making.
- Refer to the participant as "The interviewee" or "The participant".

PARTICIPANT DATA:
<PARTICIPANT_DATA_BLOCK>"""

EXPERT_PROMPTS["behavioral_economist"] = """You are an expert Behavioral Economist (with a PhD) specializing in household financial management, risk behavior, informal credit systems, and material constraints in developing economies.

TASK:
Analyze the provided participant self-report data, which contains BOTH structured survey responses (Section 1) AND an open-ended qualitative interview transcript (Section 2). Cross-reference their stated financial choices with their qualitative accounts of money management to generate a list of economic observations.

KEY FOCUS AREAS:
1. Income predictability, cash-flow friction, and material constraints.
2. Financial risk tolerance versus security-seeking behavior.
3. Utilization of formal banking versus informal financial mechanisms (e.g., local moneylenders, chit funds/seettu, gold loans, borrowing from kin).
4. Short-term spending pressure versus long-term savings and asset-building goals.
5. Monetary altruism, reciprocity, and willingness to share resources under financial stress.

SYNTHESIS INSTRUCTIONS:
- Compare stated financial attitudes/income brackets in Section 1 with actual financial choices, debts, and spending obligations described in Section 2.
- Identify any gap between stated financial willingness and actual material capacity.

OUTPUT FORMAT:
- Generate between 5 and 20 concise bullet-point economic observations.
- Highlight practical trade-offs and decision patterns.
- Refer to the participant as "The interviewee" or "The participant".

PARTICIPANT DATA:
<PARTICIPANT_DATA_BLOCK>"""

EXPERT_PROMPTS["political_scientist"] = """You are an expert Political Scientist and Public Policy Specialist (with a PhD) specializing in Tamil Nadu regional governance, welfare economics, civic engagement, and social justice.

TASK:
Analyze the provided participant self-report data, which contains BOTH structured survey responses (Section 1) AND an open-ended qualitative interview transcript (Section 2). Cross-reference their survey attitude selections with their qualitative discussions on society and government to generate a list of civic and political observations.

KEY FOCUS AREAS:
1. Civic awareness, political interest, and electoral/community participation.
2. Dependence on and attitudes toward state welfare schemes (e.g., free rations, women's assistance schemes, healthcare cards, student aid).
3. Views on social equality, regional identity, self-respect principles, and linguistic/cultural pride.
4. Perceptions of local administration, public services, political trust, and corruption.
5. General ideological balance regarding state intervention versus self-reliance.

SYNTHESIS INSTRUCTIONS:
- Use Section 1 survey selections to anchor their stated stances on policy/social issues.
- Use Section 2 interview content to uncover the personal experiences (e.g., interacting with local government offices, receiving welfare) that shaped those attitudes.

OUTPUT FORMAT:
- Generate between 5 and 20 concise bullet-point civic/political observations.
- Focus on regional social dynamics and policy realities rather than Western political labels.
- Refer to the participant as "The interviewee" or "The participant".

PARTICIPANT DATA:
<PARTICIPANT_DATA_BLOCK>"""

EXPERT_PROMPTS["sociologist"] = """You are an expert Sociologist and Cultural Anthropologist (with a PhD) specializing in South Asian kinship systems, Tamil cultural norms, multi-generational family structures, and community expectations.

TASK:
Analyze the provided participant self-report data, which contains BOTH structured survey responses (Section 1) AND an open-ended qualitative interview transcript (Section 2). Cross-reference their survey responses with their open-ended family narratives to generate a list of sociological observations.

KEY FOCUS AREAS:
1. Filial deference and parental authority in major life choices (education, career, relocation, marriage).
2. Sensitivity to family honor (*kudumba maanam*), social status, and community judgment/gossip.
3. Multi-generational obligations (elder care expectations, living with parents/in-laws).
4. Participation in social/kinship reciprocity networks (e.g., ceremonial gifts/*moi*, temple/festival duties, attending family functions).
5. Gender role dynamics within household labor and decision-making.

SYNTHESIS INSTRUCTIONS:
- Use Section 1 survey answers on family and trust as baseline indicators.
- Use Section 2 interview text to map out their exact family tree, key interpersonal conflicts, and cultural tension points between individual desires and collective obligations.

OUTPUT FORMAT:
- Generate between 5 and 20 concise bullet-point sociological observations.
- Highlight latent social pressures that constrain or dictate individual choices.
- Refer to the participant as "The interviewee" or "The participant".

PARTICIPANT DATA:
<PARTICIPANT_DATA_BLOCK>"""

EXPERT_ORDER = ["demographer", "psychologist", "behavioral_economist",
                "political_scientist", "sociologist"]

# --------------------------------------------------------------------------
# Language instructions appended to every expert prompt
# --------------------------------------------------------------------------
TAMIL_INSTRUCTIONS = [
    "Write ALL observations in Tamil (தமிழ்) script. This overrides any English "
    "wording in the instructions above, including the phrase 'The interviewee' / 'The participant'.",
    "Refer to the person as 'நேர்காணலில் பங்கேற்றவர்' (or 'பங்கேற்பாளர்'), never in English.",
    "Keep numbers, survey scores, scale names (e.g. BFI-44) and proper nouns unchanged. "
    "For a technical term you may give the English word once in brackets, "
    "e.g. புறமுகத்தன்மை (Extraversion).",
    "Quote the participant in their original Tamil. Do not write full English sentences.",
    "Output only the bullet list (each line starting with '- '), with no introduction or closing remarks.",
]

ENGLISH_INSTRUCTIONS = [
    "Write all observations in English. Keep short Tamil quotes in Tamil script if you cite them.",
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def load_env_file() -> None:
    """Tiny .env reader (no extra dependency). Existing env vars win."""
    for folder in (SCRIPT_DIR, Path.cwd()):
        path = folder / ".env"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_client():
    load_env_file()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        key = getpass("Paste your GEMINI_API_KEY (not echoed, not saved): ")
    from google import genai  # imported here so --dry-run works without the SDK
    return genai.Client(api_key=key)


def call_llm(client, prompt: str, model: str, temperature: float, retries: int = 3) -> str:
    last_err: Exception | None = None
    models = tuple(dict.fromkeys((model, *FALLBACK_MODELS)))
    for current_model in models:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config={"temperature": temperature},
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("model returned an empty response")
                return text
            except Exception as exc:  # noqa: BLE001 - retry wrapper
                last_err = exc
                error_text = str(exc)
                retryable = any(code in error_text for code in ("429", "500", "502", "503", "504"))
                if not retryable or attempt == retries - 1:
                    break
                wait = 2 ** attempt
                print(f"    {current_model} failed ({type(exc).__name__}: {exc}); retrying in {wait}s")
                time.sleep(wait)
        if current_model != models[-1]:
            print(f"    trying fallback model {models[models.index(current_model) + 1]}")
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# Loading and shaping the survey data (all.json)
# --------------------------------------------------------------------------
def load_json_tolerant(text: str):
    """Parse one JSON document, or several concatenated ones (-> list)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder, i, objs = json.JSONDecoder(), 0, []
    try:
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            obj, i = decoder.raw_decode(text, i)
            objs.append(obj)
        return objs if objs else None
    except json.JSONDecodeError:
        return None


LABEL_FIELDS = ("module", "module_name", "section", "name", "title", "id", "question_id")


def _norm(value) -> str | None:
    return value.strip().lower() if isinstance(value, str) else None


def _is_labelled_match(d: dict, names: set[str]) -> bool:
    return any(_norm(d.get(f)) in names for f in LABEL_FIELDS if f in d)


def prune(obj, names: set[str], counter: list[int]):
    """Remove dict keys / list items / labelled sub-objects whose label is in `names`."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _norm(k) in names:
                counter[0] += 1
                continue
            if isinstance(v, dict) and _is_labelled_match(v, names):
                counter[0] += 1
                continue
            out[k] = prune(v, names, counter)
        return out
    if isinstance(obj, list):
        out_list = []
        for item in obj:
            if isinstance(item, dict) and _is_labelled_match(item, names):
                counter[0] += 1
                continue
            out_list.append(prune(item, names, counter))
        return out_list
    return obj


def build_survey_text(survey_path: Path, exclude: list[str]) -> tuple[str, dict]:
    raw = survey_path.read_text(encoding="utf-8")
    data = load_json_tolerant(raw)
    info = {"file": survey_path.name, "chars_raw": len(raw), "excluded_elements": 0}
    if data is None:
        print(f"  [warn] {survey_path.name} is not valid JSON; using it as plain text.")
        if exclude:
            print("  [warn] --exclude cannot be applied to non-JSON text; nothing was removed.")
        return raw.strip(), info
    if exclude:
        names = {e.strip().lower() for e in exclude}
        counter = [0]
        data = prune(data, names, counter)
        info["excluded_elements"] = counter[0]
        print(f"  Excluded {counter[0]} survey element(s) matching: {', '.join(sorted(names))}")
        if counter[0] == 0:
            print("  [warn] --exclude matched nothing; check the spelling against all.json.")
    return json.dumps(data, ensure_ascii=False, indent=2), info


def warn_possible_pii(text: str) -> None:
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phones = re.findall(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)", text)
    if emails or phones:
        print(f"  [WARNING] possible personal identifiers in the data being sent to the API: "
              f"{len(emails)} email(s), {len(phones)} phone number(s). "
              f"Remove them (or use --exclude) before running without --dry-run.")


# --------------------------------------------------------------------------
# Prompt assembly and output checks
# --------------------------------------------------------------------------
SOURCES = ("both", "interview", "survey")

SOURCE_NOTES = {
    "both": [],
    "interview": [
        "DATA AVAILABILITY: Only Section 2 (the interview transcript) exists for this run. "
        "Section 1 is intentionally empty. Ignore every instruction above that tells you to use "
        "survey scores or survey figures; base every observation on the transcript alone. "
        "Do not invent survey numbers.",
    ],
    "survey": [
        "DATA AVAILABILITY: Only Section 1 (the structured survey responses) exists for this run. "
        "Section 2 is intentionally empty and there is NO interview. Ignore every instruction above "
        "that tells you to use interview narratives; base every observation on the survey answers alone. "
        "Do not invent life stories, quotes or events. Where you go beyond the literal answers, "
        "label it as an inference.",
    ],
}


def build_block(survey_text: str, transcript: str, source: str) -> str:
    if source == "interview":
        section1 = "(No structured survey data is provided for this run.)"
    else:
        section1 = survey_text
    if source == "survey":
        section2 = "(No interview transcript is provided for this run.)"
    else:
        section2 = transcript
    return (f"=== SECTION 1: STRUCTURED SURVEY RESPONSES ===\n{section1}\n\n"
            f"=== SECTION 2: OPEN-ENDED INTERVIEW TRANSCRIPT ===\n{section2}")


def build_prompt(expert: str, block: str, lang: str, require_evidence: bool,
                 source: str = "both") -> str:
    prompt = EXPERT_PROMPTS[expert].replace(PLACEHOLDER, block)
    extras = list(SOURCE_NOTES[source])
    extras += list(TAMIL_INSTRUCTIONS if lang == "ta" else ENGLISH_INSTRUCTIONS)
    if require_evidence:
        label = "சான்று" if lang == "ta" else "Evidence"
        extras.append(f'After each observation, add a short supporting quote from the data '
                      f'in the form ({label}: "...").')
    return prompt + "\n\nADDITIONAL INSTRUCTIONS:\n" + "\n".join(f"- {e}" for e in extras)


BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+\S")


def count_observations(text: str) -> int:
    return sum(1 for line in text.splitlines() if BULLET_RE.match(line))


def tamil_ratio(text: str) -> float:
    """Share of Tamil-script characters among Tamil + Latin letters."""
    tamil = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
    latin = sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    return tamil / (tamil + latin) if (tamil + latin) else 0.0


def resolve_participant_dir(participants_dir: Path, username: str) -> tuple[Path, str] | None:
    """Resolve a folder by name or by the username stored in its all.json report."""
    if not participants_dir.is_dir():
        return None

    exact = participants_dir / username
    if exact.is_dir():
        return exact, exact.name

    wanted = username.casefold()
    for candidate in sorted(p for p in participants_dir.iterdir() if p.is_dir()):
        if candidate.name.casefold() == wanted:
            return candidate, candidate.name
        report_file = candidate / "all.json"
        if not report_file.exists():
            continue
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and str(report.get("username", "")).casefold() == wanted:
            return candidate, candidate.name
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the 5 expert reflections for one participant (survey + interview), in Tamil by default.")
    ap.add_argument("username", help="participant folder name, e.g. Mouneeswaran")
    ap.add_argument("--participants-dir", type=Path, default=SCRIPT_DIR / "export" / "participants")
    ap.add_argument("--survey-file", default="all.json")
    ap.add_argument("--transcript-file", default="interview_transcript_ta.txt")
    ap.add_argument("--out-dir-name", default=None,
                    help="output folder name (default: expert_ta / expert_interview_only_ta, "
                         "or expert / expert_interview_only with --lang en)")
    ap.add_argument("--only", nargs="+", choices=EXPERT_ORDER, help="run only these experts")
    ap.add_argument("--exclude", nargs="+", default=[], help="survey items/modules to drop before the model sees them")
    ap.add_argument("--source", choices=SOURCES, default="both",
                    help="which data the experts see: 'both' (survey + interview, default), "
                         "'interview' (transcript only), 'survey' (all.json only)")
    ap.add_argument("--no-survey", action="store_true",
                    help="alias for --source interview")
    ap.add_argument("--lang", choices=["en", "ta"], default="ta",
                    help="language of the observations (default: ta)")
    ap.add_argument("--require-evidence", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--force", action="store_true", help="regenerate files that already exist")
    ap.add_argument("--dry-run", action="store_true", help="write prompts only; no API calls")
    args = ap.parse_args(argv)

    resolved = resolve_participant_dir(args.participants_dir, args.username)
    if resolved is None:
        available = sorted(p.name for p in args.participants_dir.iterdir() if p.is_dir()) \
            if args.participants_dir.is_dir() else []
        print(f"Participant folder not found: {args.participants_dir / args.username}")
        print(f"Available: {', '.join(available) if available else '(none found)'}")
        return 2
    pdir, participant_folder = resolved

    source = "interview" if args.no_survey else args.source
    use_survey = source in ("both", "survey")
    use_transcript = source in ("both", "interview")

    print(f"Participant: {participant_folder}")
    print(f"Language   : {'Tamil (ta)' if args.lang == 'ta' else 'English (en)'}")
    print(f"Data source: {source}")

    # ---- inputs --------------------------------------------------------
    transcript = ""
    if use_transcript:
        transcript_path = pdir / args.transcript_file
        if not transcript_path.exists():
            print(f"Missing transcript: {transcript_path}  (use --source survey to run without it)")
            return 2
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            print("Transcript file is empty.")
            return 2

    survey_text, survey_info = "", {"file": None}
    if use_survey:
        survey_path = pdir / args.survey_file
        if not survey_path.exists():
            print(f"Missing survey file: {survey_path}  (use --source interview to run without it)")
            return 2
        survey_text, survey_info = build_survey_text(survey_path, args.exclude)
    elif args.exclude:
        print("  [note] --exclude is ignored because no survey data is used in this run.")

    block = build_block(survey_text, transcript, source)
    warn_possible_pii(block)

    survey_label = f"{len(survey_text):,} chars from {survey_info['file']}" if use_survey else "(omitted)"
    transcript_label = (f"{len(transcript):,} chars (~{len(transcript.split()):,} words)"
                        if use_transcript else "(omitted)")
    print(f"  Survey section : {survey_label}")
    print(f"  Transcript     : {transcript_label}")
    if use_survey:
        preview = survey_text[:300].replace("\n", " ")
        print(f"  Survey preview : {preview}{'...' if len(survey_text) > 300 else ''}")

    # ---- output location ----------------------------------------------
    folder_by_source = {"both": "expert", "interview": "expert_interview_only",
                        "survey": "expert_survey_only"}
    out_name = args.out_dir_name or (
        folder_by_source[source] + ("_ta" if args.lang == "ta" else "")
    )
    out_dir = pdir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    experts = [e for e in EXPERT_ORDER if not args.only or e in args.only]

    client = None if args.dry_run else get_client()
    log = {
        "participant": participant_folder,
        "requested_username": args.username,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "temperature": args.temperature,
        "lang": args.lang,
        "source": source,
        "require_evidence": args.require_evidence,
        "exclude": args.exclude,
        "inputs": {
            "transcript": {"file": args.transcript_file if use_transcript else None,
                           "chars": len(transcript), "sha8": sha8(transcript) if transcript else None},
            "survey": {**survey_info, "chars_sent": len(survey_text), "sha8": sha8(survey_text) if survey_text else None},
        },
        "experts": {},
    }

    failures = 0
    for expert in experts:
        out_file = out_dir / f"{expert}.txt"
        prompt = build_prompt(expert, block, args.lang, args.require_evidence, source)

        if args.dry_run:
            dry = out_dir / "_dry_run"
            dry.mkdir(exist_ok=True)
            (dry / f"{expert}.prompt.txt").write_text(prompt, encoding="utf-8")
            print(f"  [dry-run] {expert}: prompt saved ({len(prompt):,} chars)")
            continue

        if out_file.exists() and not args.force:
            print(f"  [skip] {expert}: {out_file.name} already exists (use --force to regenerate)")
            log["experts"][expert] = {"status": "skipped_existing"}
            continue

        print(f"  [run ] {expert} ...")
        started = time.time()
        try:
            text = call_llm(client, prompt, args.model, args.temperature)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [FAIL] {expert}: {exc}")
            log["experts"][expert] = {"status": "failed", "error": str(exc)}
            continue

        n_obs = count_observations(text)
        ratio = tamil_ratio(text)
        out_file.write_text(text + "\n", encoding="utf-8")
        status = "ok"
        if n_obs < 5 or n_obs > 20:
            status = "ok_count_outside_5_20"
            print(f"  [warn] {expert}: {n_obs} observations (the prompt asks for 5-20); review the file.")
        if args.lang == "ta" and ratio < MIN_TAMIL_RATIO:
            status += "_low_tamil"
            print(f"  [warn] {expert}: only {ratio:.0%} Tamil script; "
                  f"rerun with --force --only {expert} if it came back mostly English.")
        tamil_note = f", {ratio:.0%} Tamil" if args.lang == "ta" else ""
        print(f"  [done] {expert}: {n_obs} observations{tamil_note} -> {out_file.name} "
              f"({time.time() - started:.1f}s)")
        log["experts"][expert] = {"status": status, "observations": n_obs,
                                  "tamil_ratio": round(ratio, 2),
                                  "file": out_file.name, "seconds": round(time.time() - started, 1)}

    if not args.dry_run:
        (out_dir / "_run_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nOutput folder: {out_dir}")
    else:
        print(f"\nDry run complete. Prompts are in: {out_dir / '_dry_run'}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
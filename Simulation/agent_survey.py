#!/usr/bin/env python3
"""
agent_surv.py -- let a participant's digital-twin agent answer every GSS + BFI
question (the same questions the participant answers in Wave 2) and store the
answers as JSON under the participant folder.

    python agent_surv.py Mounees --source interview
    python agent_surv.py Mounees --source survey
    python agent_surv.py Mounees --source both
    python agent_surv.py Mounees --source demographics      # baseline

Output:  export/participants/<username>/agent_answers/<condition>.json

Sources (what the agent is allowed to know):
    interview     Tamil interview transcript (+ the routed expert's reflections)
    survey        Wave-1 survey answers + demographics (+ routed reflections)
    both          transcript + Wave-1 survey (+ routed reflections)
    demographics  demographic fields only (the Park et al. baseline)

Rules that keep the test honest:
  * This script NEVER opens wave2/.  Wave 2 is ground truth and is read only
    by agent_vs_wave2.py / evaluate_all.py.
  * Survey-based agents do not see the item they are predicting.  For a BFI
    item the whole trait (all items of that trait) is hidden, as in Park et al.
  * Expert reflections are routed: each question is sent to ONE expert and only
    that expert's notes are added (Park et al.).  The router result is cached
    in question_expert_map.json, so the router costs one API call, once.
  * Items whose display_condition is not met (e.g. "ask only if working") are
    skipped for everyone.
  * The participant's username / id is never put in a prompt.

Other flags: --reflections none (raw data only), --batch-size N, --limit N,
--force (redo finished items), --dry-run (write the first prompt, no API call).
Re-running resumes where it stopped.  The API key is read from GEMINI_API_KEY
or a .env file next to this script; it is never printed or saved.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from getpass import getpass
from pathlib import Path

import twin_common as tc

SCRIPT_DIR = tc.SCRIPT_DIR
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ROUTER_CACHE = SCRIPT_DIR / "question_expert_map.json"

EXPERT_FILES = {
    "Demographer": "demographer.txt",
    "Psychologist": "psychologist.txt",
    "Behavioral Economist": "behavioral_economist.txt",
    "Regional Political Scientist": "political_scientist.txt",
    "Sociologist & Cultural Anthropologist": "sociologist.txt",
}
EXPERT_BLURBS = {
    "Demographer": "age, life stage, education, work, income, household and place",
    "Psychologist": "personality, emotions, coping, autonomy, wellbeing, self-description",
    "Behavioral Economist": "money, risk, trust, cooperation, savings, borrowing",
    "Regional Political Scientist": "politics, government, welfare, civic life, regional identity",
    "Sociologist & Cultural Anthropologist": "family, kinship, caste/community, religion, gender roles, social duty",
}
REFLECTION_DIRS = {   # (Tamil-output folder, English-output folder)
    "interview": ("expert_interview_only_ta", "expert_interview_only"),
    "survey": ("expert_survey_only_ta", "expert_survey_only"),
    "both": ("expert_ta", "expert"),
}

SECTION_RULES = {"B": "Behavioral Economist", "C": "Sociologist & Cultural Anthropologist",
                 "D": "Behavioral Economist", "E": "Sociologist & Cultural Anthropologist",
                 "F": "Regional Political Scientist", "G": "Psychologist"}
KEY_RULES = {
    "gss_q04_conpress": "Regional Political Scientist", "gss_q06_conclerg": "Regional Political Scientist",
    "gss_q07_family16": "Demographer", "gss_q15_satjob": "Demographer", "gss_q20_class": "Demographer",
    "gss_q33_local": "Sociologist & Cultural Anthropologist",
    "gss_q34_local": "Sociologist & Cultural Anthropologist",
    "gss_q39_health": "Demographer", "gss_q40_socrel": "Sociologist & Cultural Anthropologist",
    "gss_q42_local": "Sociologist & Cultural Anthropologist",
}

SYSTEM_PROMPT = """You are simulating ONE specific research participant from a study in Tamil Nadu, India.
Using ONLY the participant data provided, predict how this participant would answer each survey question the next time they are asked (about two weeks later).

Rules:
- Answer as this participant would, not as an average person and not as you would.
- Ground each answer in the supplied data (interview, survey answers, expert notes). Expert notes are interpretation, not new facts.
- If nothing speaks directly to a question, infer the most likely answer from related evidence. Never refuse and never answer "unknown".
- Single-choice questions: the answer MUST be copied exactly from the options list.
- Agree/disagree statements: the answer MUST be one integer from 1 to 5 (1 = strongly disagree, 5 = strongly agree).
- Think first: write a brief reasoning (at most 3 sentences, citing the evidence) BEFORE the answer.
- Output JSON only."""


# --------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------
def load_env_file() -> None:
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
    from google import genai  # imported late so --dry-run works without the SDK
    return genai.Client(api_key=key)


def call_model(client, model: str, system: str, user: str, temperature: float,
               retries: int = 4) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model, contents=user,
                config={"system_instruction": system, "temperature": temperature,
                        "response_mime_type": "application/json"})
            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError("empty response")
            return text
        except Exception as exc:  # noqa: BLE001 - retry wrapper
            last = exc
            rate = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            wait = 20 * (attempt + 1) if rate else 2 ** attempt
            print(f"    model call failed ({type(exc).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"model call failed after {retries} attempts: {last}")


def parse_json_loose(text: str):
    import json
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", t, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# --------------------------------------------------------------------------
# inputs: transcript, reflections, survey view
# --------------------------------------------------------------------------
def tamil_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("\u0b80" <= c <= "\u0bff" for c in letters) / len(letters) if letters else 0.0


def load_transcript(pdir: Path):
    path = pdir / "interview_transcript_ta.txt"
    if not path.exists():
        raise SystemExit(f"Missing transcript: {path}  (needed for --source interview/both)")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Transcript is empty: {path}")
    answers = re.findall(r"^Transcript:\s*(.*)$", text, flags=re.M)
    info = {"file": path.name, "chars": len(text), "sha8": tc.sha8(text),
            "answers_total": len(answers), "answers_with_text": sum(bool(a.strip()) for a in answers),
            "tamil_ratio": round(tamil_ratio(text), 2)}
    if info["tamil_ratio"] < 0.2:
        print(f"  [WARNING] transcript is only {info['tamil_ratio']:.0%} Tamil script - "
              f"is it a placeholder instead of real transcription?")
    if info["answers_total"] and info["answers_with_text"] < info["answers_total"]:
        print(f"  [warn] {info['answers_total'] - info['answers_with_text']} interview answer(s) have no transcript text.")
    return text, info


def load_reflections(pdir: Path, source: str, lang: str):
    names = REFLECTION_DIRS[source]
    for name in (names if lang == "ta" else tuple(reversed(names))):
        d = pdir / name
        if d.is_dir():
            out = {}
            for expert, fname in EXPERT_FILES.items():
                f = d / fname
                if f.is_file() and f.read_text(encoding="utf-8").strip():
                    out[expert] = f.read_text(encoding="utf-8").strip()
            return out, name
    return {}, None


def rules_route(item: tc.Item) -> str:
    if item.kind == "bfi":
        return "Psychologist"
    if item.key in KEY_RULES:
        return KEY_RULES[item.key]
    return SECTION_RULES.get(item.section[:1].upper(), "Psychologist")


def build_router(client, model: str, bank: list[tc.Item], mode: str, refresh: bool) -> dict:
    """question key -> expert name.  'llm' = one batched call, cached; 'rules' = fixed mapping."""
    rules = {i.key: rules_route(i) for i in bank}
    if mode == "rules" or client is None:
        return rules
    bank_hash = tc.sha8("|".join(i.key + i.question for i in bank))
    if ROUTER_CACHE.exists() and not refresh:
        cache = tc.read_json(ROUTER_CACHE)
        if cache.get("bank_sha8") == bank_hash:
            return {**rules, **{k: v for k, v in cache["routes"].items() if v in EXPERT_FILES}}
    print("  Building question->expert router (one API call, cached) ...")
    listing = "\n".join(f"{i.key}: {i.question}" for i in bank)
    experts = "\n".join(f"- {n}: {b}" for n, b in EXPERT_BLURBS.items())
    user = (f"Experts:\n{experts}\n\nFor each survey question choose the ONE expert whose notes would "
            f"best help predict a person's answer.\n\nQuestions:\n{listing}\n\n"
            'Return JSON: {"routes": {"<key>": "<exact expert name>", ...}}')
    try:
        obj = parse_json_loose(call_model(client, model, "You route survey questions to experts.", user, 0.0))
        routes = {k: v for k, v in obj.get("routes", {}).items() if k in rules and v in EXPERT_FILES}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] LLM routing failed ({exc}); using the fixed rules instead.")
        return rules
    missing = [k for k in rules if k not in routes]
    if missing:
        print(f"  [warn] router skipped {len(missing)} question(s); fixed rules used for them.")
    tc.write_json(ROUTER_CACHE, {"bank_sha8": bank_hash, "model": model, "created_at": tc.now_iso(),
                                 "routes": {**rules, **routes}})
    return {**rules, **routes}


def render_survey_view(wave1: tc.Wave, bank: list[tc.Item], masked: set[str],
                       demographics_only: bool = False) -> str:
    lines = []
    if wave1.demographics:
        lines.append("DEMOGRAPHICS")
        lines += [f"- {k}: {v}" for k, v in wave1.demographics.items()]
    if demographics_only:
        return "\n".join(lines)
    section = None
    lines.append("\nWAVE-1 SURVEY ANSWERS (questions you are asked to predict are hidden)")
    for item in bank:
        if item.key in masked:
            continue
        value = tc.canon_answer(item, wave1.raw_answer(item))
        if value is None:
            continue
        if item.section != section:
            section = item.section
            lines.append(f"\n[{section}]")
        if item.kind == "gss":
            lines.append(f"Q: {item.question}\nA: {item.options[value - 1]}")
        else:
            lines.append(f'- "{item.question}": {value} ({tc.LIKERT_LABELS[value - 1]})')
    for g in wave1.games:
        ans = g.get("answer")
        if tc.is_blank_answer(ans):
            continue
        scenario = re.split(r"\r?\n\r?\n", str(g.get("question", "")).strip())[0][:300]
        lines.append(f"\n[{g.get('section', 'Game')}] {scenario}\nAnswer: {ans}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# prompt and parsing
# --------------------------------------------------------------------------
def render_questions(chunk: list[tc.Item]) -> str:
    out = []
    for n, it in enumerate(chunk, 1):
        if it.qtype == "likert5":
            out.append(f'[{n}] key: {it.key}\n    Statement: "{it.question}"\n'
                       f"    How well does it describe the participant? Integer 1-5 "
                       f"(1 strongly disagree, 2 disagree, 3 neither, 4 agree, 5 strongly agree).")
        else:
            opts = "\n".join(f"      - {o}" for o in it.options)
            out.append(f"[{n}] key: {it.key}\n    Question: {it.question}\n"
                       f"    Options (choose exactly one, copy the text exactly):\n{opts}")
    return "\n\n".join(out)


def build_user_prompt(source: str, transcript: str | None, expert: str | None,
                      reflection: str | None, survey_view: str | None,
                      chunk: list[tc.Item], correction: str = "") -> str:
    parts = ["PARTICIPANT DATA"]
    if transcript:
        parts.append(f"=== INTERVIEW TRANSCRIPT (Tamil) ===\n{transcript}")
    if reflection:
        parts.append(f"=== EXPERT NOTES: {expert} ===\n{reflection}")
    if survey_view:
        parts.append(f"=== {'DEMOGRAPHICS' if source == 'demographics' else 'SURVEY DATA'} ===\n{survey_view}")
    parts.append("QUESTIONS TO ANSWER\n" + render_questions(chunk))
    parts.append('Return ONLY JSON, reasoning before answer:\n'
                 '{"answers": [{"key": "<key>", "reasoning": "<=3 sentences>", "answer": <option text or integer>}]}')
    if correction:
        parts.append(f"CORRECTION: {correction}")
    return "\n\n".join(parts)


def extract_entries(obj) -> dict:
    if isinstance(obj, dict) and isinstance(obj.get("answers"), list):
        obj = obj["answers"]
    if isinstance(obj, list):
        return {str(e["key"]): e for e in obj if isinstance(e, dict) and "key" in e}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if isinstance(v, dict)}
    return {}


def make_record(item: tc.Item, value: int, entry: dict, expert, masked: set[str], attempts: int) -> dict:
    rec = {"type": item.qtype, "status": "ok", "expert": expert, "attempts": attempts,
           "reasoning": str(entry.get("reasoning", ""))[:700], "masked_keys": sorted(masked)}
    if item.qtype == "likert5":
        rec.update(answer=value, answer_index=None, answer_label=tc.LIKERT_LABELS[value - 1])
    else:
        rec.update(answer=item.options[value - 1], answer_index=value)
    return rec


def chunk_items(todo: list[tc.Item], routes: dict, size: int) -> list[list[tc.Item]]:
    chunks, cur, cur_key = [], [], None
    for it in todo:
        k = (it.kind, it.section, routes.get(it.key))
        if cur and (k != cur_key or len(cur) >= size):
            chunks.append(cur)
            cur = []
        cur.append(it)
        cur_key = k
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a participant's agent over the GSS + BFI questions.")
    ap.add_argument("username", help="participant folder name, e.g. Mounees")
    ap.add_argument("--source", choices=["interview", "survey", "both", "demographics"], default="both")
    ap.add_argument("--reflections", choices=["routed", "none"], default="routed")
    ap.add_argument("--reflection-lang", choices=["ta", "en"], default="ta",
                    help="which reflection folder to prefer (default: *_ta)")
    ap.add_argument("--questions", type=Path, default=tc.DEFAULT_QUESTIONS_FILE)
    ap.add_argument("--participants-dir", type=Path, default=tc.DEFAULT_PARTICIPANTS_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=1, help="questions per call (1 = item by item, as in the paper)")
    ap.add_argument("--router", choices=["llm", "rules"], default="llm")
    ap.add_argument("--refresh-router", action="store_true")
    ap.add_argument("--only-kind", choices=["gss", "bfi"])
    ap.add_argument("--limit", type=int, help="only the first N questions (quick test)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between calls")
    ap.add_argument("--force", action="store_true", help="redo items that are already answered")
    ap.add_argument("--dry-run", action="store_true", help="write the first prompt, make no API call")
    args = ap.parse_args(argv)

    pdir = tc.find_participant_dir(args.username, args.participants_dir)
    bank = tc.load_question_bank(args.questions)
    wave1 = tc.load_wave(pdir / "all.json")        # Wave 1 only - this script never opens wave2/
    use_reflections = args.reflections == "routed" and args.source != "demographics"
    condition = args.source if (args.source == "demographics" or use_reflections) else f"{args.source}_noref"

    print(f"Participant folder: {pdir.name}   condition: {condition}   model: {args.model}")

    transcript, tinfo, inputs = None, None, []
    if args.source in ("interview", "both"):
        transcript, tinfo = load_transcript(pdir)
        inputs.append({"file": tinfo["file"], "chars": tinfo["chars"], "sha8": tinfo["sha8"]})
        print(f"  Transcript: {tinfo['chars']:,} chars, {tinfo['answers_with_text']}/{tinfo['answers_total']} "
              f"answers with text, {tinfo['tamil_ratio']:.0%} Tamil script")
    if args.source in ("survey", "both", "demographics"):
        inputs.append({"file": "all.json", "chars": len(str(wave1.raw)), "sha8": tc.sha8(str(wave1.raw))})

    reflections, refl_dir = ({}, None)
    if use_reflections:
        reflections, refl_dir = load_reflections(pdir, args.source, args.reflection_lang)
        if not reflections:
            print(f"  [warn] no reflection folder/files found for source '{args.source}' "
                  f"({' or '.join(REFLECTION_DIRS[args.source])}); continuing without reflections.")
        else:
            inputs += [{"file": f"{refl_dir}/{EXPERT_FILES[e]}"} for e in reflections]
            print(f"  Reflections: {refl_dir} ({len(reflections)} experts, routed per question)")

    notes = []
    if args.source in ("survey", "both") and reflections:
        notes.append("Reflections were generated from the FULL Wave-1 survey and may reveal the masked answers. "
                     "Treat these results as a retest-style upper bound; re-run with --reflections none for a clean test.")
    if args.source in ("interview", "both"):
        notes.append("Interview content that overlaps with GSS topics is not filtered out.")
    if args.source in ("survey", "both") and reflections:
        print(f"  [WARNING] {notes[0]}")

    client = None if args.dry_run else get_client()
    routes = build_router(client, args.model, bank, args.router, args.refresh_router) if reflections else {}

    # ---- decide what to ask ------------------------------------------------
    out_path = pdir / tc.AGENT_DIRNAME / f"{condition}.json"
    result = {"answers": {}}
    if args.dry_run:
        args.force = True            # show the first prompt even if answers already exist; never write answers
    if out_path.exists() and not args.force:
        result = tc.read_json(out_path)
        if result.get("model") and result["model"] != args.model:
            print(f"  [warn] existing answers came from model '{result['model']}', this run uses '{args.model}'. "
                  f"Use --force to redo everything with one model.")
    result.update({"participant_folder": pdir.name, "condition": condition, "source": args.source,
                   "reflections": "routed" if reflections else "none", "model": args.model,
                   "temperature": args.temperature, "batch_size": args.batch_size,
                   "inputs_used": inputs, "notes": notes,
                   "leakage_risk": bool(args.source in ("survey", "both") and reflections),
                   "created_at": result.get("created_at", tc.now_iso())})
    answers = result["answers"]

    candidates = [i for i in bank if not args.only_kind or i.kind == args.only_kind]
    if args.limit:
        candidates = candidates[: args.limit]
    todo = []
    for it in candidates:
        if not tc.applicable(it, wave1.demographics):
            answers[it.key] = {"status": "not_applicable", "type": it.qtype}
            continue
        if not args.force and answers.get(it.key, {}).get("status") == "ok":
            continue
        todo.append(it)

    print(f"  Questions: {len(candidates)} selected, {len(todo)} to ask "
          f"({sum(a.get('status') == 'not_applicable' for a in answers.values())} not applicable)")

    chunks = chunk_items(todo, routes, max(1, args.batch_size))
    failures = done = 0
    for ci, chunk in enumerate(chunks, 1):
        expert = routes.get(chunk[0].key) if reflections else None
        masked = set()
        for it in chunk:
            masked |= tc.mask_keys_for(it, bank)
        survey_view = None
        if args.source in ("survey", "both"):
            survey_view = render_survey_view(wave1, bank, masked)
        elif args.source == "demographics":
            survey_view = render_survey_view(wave1, bank, set(), demographics_only=True)
        refl_text = reflections.get(expert) if expert else None

        correction, entries = "", {}
        for attempt in range(1, 4):
            user = build_user_prompt(args.source, transcript, expert, refl_text, survey_view, chunk, correction)
            if args.dry_run:
                p = pdir / tc.AGENT_DIRNAME / f"_dry_run_{condition}.txt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("=== SYSTEM ===\n" + SYSTEM_PROMPT + "\n\n=== USER ===\n" + user, encoding="utf-8")
                print(f"  [dry-run] first prompt written to {p} ({len(user):,} chars)")
                return 0
            try:
                entries = extract_entries(parse_json_loose(call_model(client, args.model, SYSTEM_PROMPT, user, args.temperature)))
            except Exception as exc:  # noqa: BLE001
                entries, correction = {}, f"Your previous output was not valid JSON ({type(exc).__name__}). Return only the JSON object."
                continue
            bad = [it.key for it in chunk if tc.canon_answer(it, entries.get(it.key, {}).get("answer")) is None]
            if not bad:
                break
            correction = ("These answers were missing or not one of the allowed options: "
                          + ", ".join(bad) + ". Answer them again exactly as specified.")
        for it in chunk:
            entry = entries.get(it.key, {})
            value = tc.canon_answer(it, entry.get("answer"))
            if value is None:
                failures += 1
                answers[it.key] = {"status": "failed", "type": it.qtype, "expert": expert, "attempts": attempt,
                                   "raw_answer": str(entry.get("answer"))[:200]}
                print(f"  [FAIL] {it.key}")
            else:
                answers[it.key] = make_record(it, value, entry, expert, tc.mask_keys_for(it, bank), attempt)
                done += 1
                print(f"  [{done + failures:3d}/{len(todo)}] {it.key} -> {answers[it.key]['answer']}")
        result["updated_at"] = tc.now_iso()
        tc.write_json(out_path, result)          # progress is saved after every call
        if args.sleep:
            time.sleep(args.sleep)

    result["updated_at"] = tc.now_iso()
    if not args.dry_run:
        tc.write_json(out_path, result)
    n_ok = sum(a.get("status") == "ok" for a in answers.values())
    print(f"\nSaved: {out_path}\n  ok={n_ok}  failed={sum(a.get('status') == 'failed' for a in answers.values())}  "
          f"not_applicable={sum(a.get('status') == 'not_applicable' for a in answers.values())}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
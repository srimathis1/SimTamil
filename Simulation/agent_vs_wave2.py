#!/usr/bin/env python3
"""
agent_vs_wave2.py -- compare every agent's answers with the participant's real
Wave-2 answers, item by item, and report accuracy.

    python agent_vs_wave2.py Mounees
    python agent_vs_wave2.py Mounees --conditions interview both

Reads (never writes) : <participant>/all.json                 Wave 1
                       <participant>/wave2/wave2.json          Wave 2 (ground truth)
                       <participant>/agent_answers/*.json      from agent_surv.py
Writes               : <participant>/agent_eval/comparison_<condition>.json
                       <participant>/agent_eval/item_universe.json

A "human_wave1" row is always added: the participant's own Wave-1 answer used
as the prediction of Wave 2.  That is the test-retest ceiling -- an agent
cannot be expected to beat the person's own consistency, and
evaluate_all.py divides by it to get normalized accuracy.

Items that are scored (the same set for every condition, so they are
comparable):  applicable to the participant (display_condition) AND answered
in Wave 1 AND answered in Wave 2.  If the agent failed to answer a scored
item, it counts as wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import twin_common as tc


def build_universe(bank: list[tc.Item], wave1: tc.Wave, wave2: tc.Wave) -> dict:
    scored, excluded, wave2_unapplicable = [], [], []
    for item in bank:
        w2 = tc.canon_answer(item, wave2.raw_answer(item))
        if not tc.applicable(item, wave1.demographics):
            excluded.append({"key": item.key, "reason": "not_applicable"})
            if w2 is not None:
                wave2_unapplicable.append(item.key)
            continue
        w1 = tc.canon_answer(item, wave1.raw_answer(item))
        if w2 is None:
            excluded.append({"key": item.key, "reason": "no_wave2_answer"})
        elif w1 is None:
            excluded.append({"key": item.key, "reason": "no_wave1_answer"})
        else:
            scored.append({"item": item, "w1": w1, "w2": w2})
    return {"scored": scored, "excluded": excluded, "wave2_unapplicable": wave2_unapplicable}


def load_agent_values(path: Path, bank_by_key: dict) -> tuple[dict, dict]:
    """-> ({key: canonical value or None}, metadata)"""
    data = tc.read_json(path)
    values = {}
    for key, rec in data.get("answers", {}).items():
        item = bank_by_key.get(key)
        if item is None or rec.get("status") != "ok":
            continue
        values[key] = rec.get("answer_index") if item.qtype == "single_choice" else rec.get("answer")
    return values, data


def make_records(scored: list[dict], values: dict) -> list[dict]:
    records = []
    for s in scored:
        item, w2 = s["item"], s["w2"]
        pred = values.get(item.key)
        failed = pred is None
        exact = (not failed) and pred == w2
        dist = None if failed else tc.ordinal_distance(item, pred, w2)
        records.append({
            "key": item.key, "kind": item.kind, "section": item.section, "trait": item.trait,
            "reverse": item.reverse, "n_options": len(item.options) if item.kind == "gss" else 5,
            "question": item.question,
            "wave1": s["w1"], "wave2": w2, "pred": pred,
            "wave1_text": tc.value_text(item, s["w1"]), "wave2_text": tc.value_text(item, w2),
            "pred_text": tc.value_text(item, pred),
            "failed": failed, "exact": exact,
            "ordinal_distance": dist,
            "within1": (dist <= 1) if dist is not None else None,
        })
    return records


def summarize(records: list[dict]) -> dict:
    def acc(rs):
        return round(sum(r["exact"] for r in rs) / len(rs), 4) if rs else None
    gss = [r for r in records if r["kind"] == "gss"]
    bfi = [r for r in records if r["kind"] == "bfi"]
    return {"n_scored": len(records), "n_gss": len(gss), "n_bfi": len(bfi),
            "n_failed": sum(r["failed"] for r in records),
            "gss_exact": acc(gss), "bfi_exact": acc(bfi), "overall_exact": acc(records),
            "bfi_within1": (round(sum(bool(r["within1"]) for r in bfi) / len(bfi), 4) if bfi else None)}


def compare_all(pdir: Path, bank: list[tc.Item], conditions: list[str] | None = None,
                write: bool = True) -> dict:
    wave1 = tc.load_wave(pdir / "all.json")
    w2_path = pdir / "wave2" / "wave2.json"
    if not w2_path.exists():
        raise SystemExit(f"No Wave-2 file for this participant: {w2_path}")
    wave2 = tc.load_wave(w2_path)
    bank_by_key = {i.key: i for i in bank}
    universe = build_universe(bank, wave1, wave2)
    scored = universe["scored"]

    agent_dir = pdir / tc.AGENT_DIRNAME
    files = sorted(p for p in agent_dir.glob("*.json") if not p.name.startswith("_")) if agent_dir.is_dir() else []
    if conditions:
        files = [p for p in files if p.stem in conditions]
        missing = set(conditions) - {p.stem for p in files}
        if missing:
            print(f"  [warn] no agent answers found for: {', '.join(sorted(missing))}")

    results = {}
    human_values = {s["item"].key: s["w1"] for s in scored}
    recs = make_records(scored, human_values)
    results[tc.HUMAN_CONDITION] = {"meta": {"condition": tc.HUMAN_CONDITION, "leakage_risk": False},
                                   "records": recs, "summary": summarize(recs)}
    for path in files:
        values, meta = load_agent_values(path, bank_by_key)
        recs = make_records(scored, values)
        results[path.stem] = {
            "meta": {k: meta.get(k) for k in ("condition", "source", "reflections", "model", "temperature",
                                              "batch_size", "leakage_risk", "notes", "updated_at")},
            "records": recs, "summary": summarize(recs)}

    if write:
        out = pdir / tc.EVAL_DIRNAME
        for cond, res in results.items():
            tc.write_json(out / f"comparison_{cond}.json", {"participant_folder": pdir.name, **res})
        tc.write_json(out / "item_universe.json", {
            "scored_keys": [s["item"].key for s in scored],
            "excluded": universe["excluded"],
            "wave2_answered_but_not_applicable": universe["wave2_unapplicable"]})
    return {"universe": universe, "results": results}


def print_table(results: dict) -> None:
    print(f"\n{'condition':<18}{'n':>4}{'GSS exact':>11}{'BFI exact':>11}{'BFI +/-1':>10}{'all exact':>11}{'failed':>8}")
    for cond, res in results.items():
        s = res["summary"]
        f = lambda v: "  -  " if v is None else f"{v:.1%}"  # noqa: E731
        print(f"{cond:<18}{s['n_scored']:>4}{f(s['gss_exact']):>11}{f(s['bfi_exact']):>11}"
              f"{f(s['bfi_within1']):>10}{f(s['overall_exact']):>11}{s['n_failed']:>8}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare agent answers with Wave 2.")
    ap.add_argument("username")
    ap.add_argument("--conditions", nargs="+", help="only these agent conditions (file names in agent_answers/)")
    ap.add_argument("--questions", type=Path, default=tc.DEFAULT_QUESTIONS_FILE)
    ap.add_argument("--participants-dir", type=Path, default=tc.DEFAULT_PARTICIPANTS_DIR)
    args = ap.parse_args(argv)

    pdir = tc.find_participant_dir(args.username, args.participants_dir)
    bank = tc.load_question_bank(args.questions)
    out = compare_all(pdir, bank, args.conditions)
    u = out["universe"]

    print(f"Participant folder: {pdir.name}")
    print(f"Scored items: {len(u['scored'])}  (excluded: {len(u['excluded'])} -> "
          f"{', '.join(sorted({e['reason'] for e in u['excluded']})) or 'none'})")
    if u["wave2_unapplicable"]:
        print(f"  [note] Wave 2 contains answers to items that were NOT applicable per Wave-1 demographics: "
              f"{', '.join(u['wave2_unapplicable'])}. The Wave-2 form ignores display_condition; "
              f"these items are left out of scoring.")
    if len(out["results"]) == 1:
        print("  [warn] no agent answer files found in agent_answers/ - only the human baseline is shown.")
    print_table(out["results"])
    print(f"\nSaved to: {pdir / tc.EVAL_DIRNAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
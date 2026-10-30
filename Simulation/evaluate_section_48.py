#!/usr/bin/env python3
"""
evaluate_section_48.py -- Computes Sections 4.8.1 to 4.8.4 per-participant and
stores results in export/evaluation_scores/.

Metrics implemented:
  4.8.1  Human Test-Retest Consistency (C):
         C = (1 / (N * M)) * sum( I(Y_wave1 == Y_wave2) )
         Computed from comparison_human_wave1.json (wave1 vs wave2 exact matches).

  4.8.2  Agent Response Agreement Rate (A):
         A = (1 / (N * M)) * sum( I(Y_hat == Y_wave2) ) * 100
         Computed from comparison_both.json (agent 'both' condition vs wave2).

  4.8.3  Normalized Accuracy (A_n):
         A_n = A / (100 * C)
         Values >= 1 mean agent matches or beats human test-retest ceiling.

  4.8.4  Mean Absolute Error (MAE) on ordinal values:
         MAE = (1 / N) * sum( |y - y_hat| )
         Computed for BFI items (1-5 Likert) and GSS items separately,
         using the ordinal_distance field from the comparison JSON.

Outputs:
  export/evaluation_scores/section_48_metrics.json
  export/evaluation_scores/section_48_metrics.csv
  export/evaluation_scores/section_48_metrics_report.md
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_PROCESSOR_DIR = Path(r"c:\Users\mayil\Music\Data_Processor")
PARTICIPANTS_DIR = DATA_PROCESSOR_DIR / "export" / "participants"
OUTPUT_DIR = DATA_PROCESSOR_DIR / "export" / "evaluation_scores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTICIPANTS = ["Mounees", "Rahul", "Srimathi"]


def load_comparison(participant: str, condition: str) -> Optional[Dict]:
    """Load a comparison JSON (human_wave1 or both)."""
    path = PARTICIPANTS_DIR / participant / "agent_eval" / f"comparison_{condition}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_human_consistency(records: List[Dict]) -> Dict[str, Any]:
    """
    4.8.1 Human Test-Retest Consistency (C).
    Uses wave1 vs wave2 exact match from comparison_human_wave1.json.
    C = (1 / M) * sum( I(wave1 == wave2) )   (per participant, N=1)
    Returns breakdown for ALL, GSS-only, BFI-only.
    """
    def _compute(recs):
        if not recs:
            return {"n": 0, "matches": 0, "C": None}
        matches = sum(1 for r in recs if r.get("wave1") == r.get("wave2"))
        C = matches / len(recs)
        return {"n": len(recs), "matches": matches, "C": round(C, 6)}

    all_recs = [r for r in records if not r.get("failed")]
    gss_recs = [r for r in all_recs if r.get("kind") == "gss"]
    bfi_recs = [r for r in all_recs if r.get("kind") == "bfi"]

    return {
        "all": _compute(all_recs),
        "gss": _compute(gss_recs),
        "bfi": _compute(bfi_recs),
    }


def compute_agent_agreement(records: List[Dict]) -> Dict[str, Any]:
    """
    4.8.2 Agent Response Agreement Rate (A).
    A = (1 / M) * sum( I(Y_hat == Y_wave2) ) * 100   (in percent)
    Uses pred vs wave2 from comparison_both.json.
    """
    def _compute(recs):
        if not recs:
            return {"n": 0, "matches": 0, "A": None}
        matches = sum(1 for r in recs if r.get("exact") is True)
        A = (matches / len(recs)) * 100.0
        return {"n": len(recs), "matches": matches, "A": round(A, 4)}

    all_recs = [r for r in records if not r.get("failed")]
    gss_recs = [r for r in all_recs if r.get("kind") == "gss"]
    bfi_recs = [r for r in all_recs if r.get("kind") == "bfi"]

    return {
        "all": _compute(all_recs),
        "gss": _compute(gss_recs),
        "bfi": _compute(bfi_recs),
    }


def compute_normalized_accuracy(A: Optional[float], C: Optional[float]) -> Optional[float]:
    """
    4.8.3 Normalized Accuracy A_n = A / (100 * C).
    A is in %, C is a proportion [0, 1].
    """
    if A is None or C is None or C == 0:
        return None
    return round(A / (100.0 * C), 6)


def compute_mae(records: List[Dict]) -> Dict[str, Any]:
    """
    4.8.4 Mean Absolute Error.
    MAE = (1 / N) * sum( |wave2 - pred| )
    Uses ordinal_distance = abs(wave2 - pred) already pre-computed.
    """
    def _compute(recs):
        valid = [r for r in recs if not r.get("failed")
                 and r.get("ordinal_distance") is not None
                 and r.get("wave2") is not None
                 and r.get("pred") is not None]
        if not valid:
            return {"n": 0, "MAE": None}
        total_abs_err = sum(abs(r["wave2"] - r["pred"]) for r in valid)
        MAE = total_abs_err / len(valid)
        return {"n": len(valid), "total_abs_error": round(total_abs_err, 4), "MAE": round(MAE, 6)}

    all_recs = records
    gss_recs = [r for r in records if r.get("kind") == "gss"]
    bfi_recs = [r for r in records if r.get("kind") == "bfi"]

    return {
        "all": _compute(all_recs),
        "gss": _compute(gss_recs),
        "bfi": _compute(bfi_recs),
    }


def compute_human_mae(records: List[Dict]) -> Dict[str, Any]:
    """
    4.8.4 Human MAE baseline: |wave1 - wave2| for each item.
    Uses records from comparison_human_wave1.json.
    """
    def _compute(recs):
        valid = [r for r in recs if not r.get("failed")
                 and r.get("wave1") is not None
                 and r.get("wave2") is not None]
        if not valid:
            return {"n": 0, "MAE": None}
        total_abs_err = sum(abs(r["wave1"] - r["wave2"]) for r in valid)
        MAE = total_abs_err / len(valid)
        return {"n": len(valid), "total_abs_error": round(total_abs_err, 4), "MAE": round(MAE, 6)}

    all_recs = records
    gss_recs = [r for r in records if r.get("kind") == "gss"]
    bfi_recs = [r for r in records if r.get("kind") == "bfi"]

    return {
        "all": _compute(all_recs),
        "gss": _compute(gss_recs),
        "bfi": _compute(bfi_recs),
    }


def evaluate_participant(username: str) -> Optional[Dict]:
    """Run all 4.8.x evaluations for a single participant."""
    human_data = load_comparison(username, "human_wave1")
    agent_data = load_comparison(username, "both")

    if not human_data:
        print(f"  [SKIP] {username}: No comparison_human_wave1.json found.")
        return None
    if not agent_data:
        print(f"  [SKIP] {username}: No comparison_both.json found.")
        return None

    h_records = human_data.get("records", [])
    a_records = agent_data.get("records", [])

    # 4.8.1 Human Test-Retest Consistency (C)
    human_consistency = compute_human_consistency(h_records)

    # 4.8.2 Agent Response Agreement Rate (A)
    agent_agreement = compute_agent_agreement(a_records)

    # 4.8.3 Normalized Accuracy (A_n)
    normalized_acc = {
        scope: compute_normalized_accuracy(
            agent_agreement[scope]["A"],
            human_consistency[scope]["C"]
        )
        for scope in ["all", "gss", "bfi"]
    }

    # 4.8.4 MAE (Agent vs Wave2)
    agent_mae = compute_mae(a_records)

    # 4.8.4 MAE (Human W1 vs W2 baseline)
    human_mae = compute_human_mae(h_records)

    result = {
        "participant": username,
        "condition": agent_data.get("meta", {}).get("condition", "both"),
        "model": agent_data.get("meta", {}).get("model", "unknown"),
        "n_scored": len([r for r in a_records if not r.get("failed")]),
        "section_481_human_consistency": human_consistency,
        "section_482_agent_agreement": agent_agreement,
        "section_483_normalized_accuracy": normalized_acc,
        "section_484_agent_mae": agent_mae,
        "section_484_human_mae_baseline": human_mae,
    }

    return result


def generate_csv_row(p: Dict) -> Dict:
    """Flatten a participant result into a CSV-friendly row."""
    def safe(d, key, sub="all"):
        try:
            return d[sub][key]
        except (KeyError, TypeError):
            return ""

    row = {
        "Participant": p["participant"],
        "Model": p["model"],
        "N_scored": p["n_scored"],
        # 4.8.1
        "Human_C_all": safe(p["section_481_human_consistency"], "C"),
        "Human_C_gss": safe(p["section_481_human_consistency"], "C", "gss"),
        "Human_C_bfi": safe(p["section_481_human_consistency"], "C", "bfi"),
        # 4.8.2
        "Agent_A_all_pct": safe(p["section_482_agent_agreement"], "A"),
        "Agent_A_gss_pct": safe(p["section_482_agent_agreement"], "A", "gss"),
        "Agent_A_bfi_pct": safe(p["section_482_agent_agreement"], "A", "bfi"),
        # 4.8.3
        "Normalized_Acc_all": p["section_483_normalized_accuracy"]["all"],
        "Normalized_Acc_gss": p["section_483_normalized_accuracy"]["gss"],
        "Normalized_Acc_bfi": p["section_483_normalized_accuracy"]["bfi"],
        # 4.8.4 agent MAE
        "Agent_MAE_all": safe(p["section_484_agent_mae"], "MAE"),
        "Agent_MAE_gss": safe(p["section_484_agent_mae"], "MAE", "gss"),
        "Agent_MAE_bfi": safe(p["section_484_agent_mae"], "MAE", "bfi"),
        # 4.8.4 human MAE baseline
        "Human_MAE_baseline_all": safe(p["section_484_human_mae_baseline"], "MAE"),
        "Human_MAE_baseline_gss": safe(p["section_484_human_mae_baseline"], "MAE", "gss"),
        "Human_MAE_baseline_bfi": safe(p["section_484_human_mae_baseline"], "MAE", "bfi"),
    }
    return row


def generate_markdown_report(results: List[Dict], path: Path) -> None:
    lines = [
        "# Section 4.8: Evaluation Metrics Report",
        "",
        f"**Generated At:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Participants:** {', '.join(p['participant'] for p in results)}",
        f"**Agent Condition:** both (Interview + Survey Wave-1 + Expert Reflections)",
        "",
        "---",
        "",
        "## 4.8.1 - Human Test-Retest Consistency (C)",
        "",
        "> $C = \\frac{1}{NM} \\sum \\mathbb{1}(Y_{\\text{w1}} = Y_{\\text{w2}})$",
        "",
        "Measures how consistently human participants answer the same questions across two waves (14-day gap).",
        "This is the **empirical ceiling** - an agent cannot reasonably be expected to exceed this.",
        "",
        "| Participant | C (All) | C (GSS only) | C (BFI only) | N items |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for p in results:
        hc = p["section_481_human_consistency"]
        lines.append(
            f"| **{p['participant']}** | "
            f"{hc['all']['C']:.4f} ({hc['all']['matches']}/{hc['all']['n']}) | "
            f"{hc['gss']['C']:.4f} ({hc['gss']['matches']}/{hc['gss']['n']}) | "
            f"{hc['bfi']['C']:.4f} ({hc['bfi']['matches']}/{hc['bfi']['n']}) | "
            f"{hc['all']['n']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4.8.2 - Agent Response Agreement Rate (A)",
        "",
        "> $A = \\frac{1}{NM} \\sum \\mathbb{1}(\\hat{Y} = Y_{\\text{w2}}) \\times 100$",
        "",
        "Percentage of agent-predicted responses that **exactly match** the participant's Wave-2 answer.",
        "",
        "| Participant | A% (All) | A% (GSS only) | A% (BFI only) | N items |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for p in results:
        aa = p["section_482_agent_agreement"]
        lines.append(
            f"| **{p['participant']}** | "
            f"{aa['all']['A']:.2f}% ({aa['all']['matches']}/{aa['all']['n']}) | "
            f"{aa['gss']['A']:.2f}% ({aa['gss']['matches']}/{aa['gss']['n']}) | "
            f"{aa['bfi']['A']:.2f}% ({aa['bfi']['matches']}/{aa['bfi']['n']}) | "
            f"{aa['all']['n']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4.8.3 - Normalized Accuracy (A_n)",
        "",
        "> $A_n = \\frac{A}{100 \\cdot C}$",
        "",
        "Compares agent accuracy against human test-retest ceiling.",
        "- **A_n = 1.0** -> Agent matches human consistency exactly",
        "- **A_n > 1.0** -> Agent exceeds human consistency",
        "- **A_n < 1.0** -> Agent below human consistency",
        "",
        "| Participant | A_n (All) | A_n (GSS) | A_n (BFI) | Interpretation |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]
    for p in results:
        na = p["section_483_normalized_accuracy"]

        def interp(v):
            if v is None:
                return "N/A"
            if v >= 1.0:
                return "Exceeds / Matches human"
            elif v >= 0.8:
                return "Near human ceiling"
            elif v >= 0.6:
                return "Moderate (60-80%)"
            else:
                return "Below 60%"

        lines.append(
            f"| **{p['participant']}** | "
            f"{na['all']:.4f} | "
            f"{na['gss']:.4f} | "
            f"{na['bfi']:.4f} | "
            f"{interp(na['all'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4.8.4 - Mean Absolute Error (MAE)",
        "",
        "> $\\text{MAE} = \\frac{1}{N} \\sum |y - \\hat{y}|$",
        "",
        "Average ordinal distance between predicted and actual responses.",
        "Also includes human W1->W2 MAE as a baseline.",
        "",
        "### Agent MAE (Agent vs Wave-2)",
        "",
        "| Participant | MAE (All) | MAE (GSS) | MAE (BFI) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for p in results:
        am = p["section_484_agent_mae"]
        lines.append(
            f"| **{p['participant']}** | "
            f"{am['all']['MAE']:.4f} | "
            f"{am['gss']['MAE']:.4f} | "
            f"{am['bfi']['MAE']:.4f} |"
        )

    lines += [
        "",
        "### Human MAE Baseline (Wave-1 vs Wave-2)",
        "",
        "| Participant | MAE Baseline (All) | MAE Baseline (GSS) | MAE Baseline (BFI) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for p in results:
        hm = p["section_484_human_mae_baseline"]
        lines.append(
            f"| **{p['participant']}** | "
            f"{hm['all']['MAE']:.4f} | "
            f"{hm['gss']['MAE']:.4f} | "
            f"{hm['bfi']['MAE']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Analytical Inferences",
        "",
        "### Human Consistency Baseline (Section 4.8.1)",
        "The three participants showed human test-retest consistency (C) ranging across items.",
        "This reflects natural human self-report variability over a 14-day period,",
        "establishing an empirical ceiling that agents can be benchmarked against.",
        "",
        "### Agent vs Human Agreement (Section 4.8.2 & 4.8.3)",
    ]

    for p in results:
        aa = p["section_482_agent_agreement"]
        hc = p["section_481_human_consistency"]
        na = p["section_483_normalized_accuracy"]
        am = p["section_484_agent_mae"]
        hm = p["section_484_human_mae_baseline"]
        lines.append(
            f"- **{p['participant']}**: A = {aa['all']['A']:.2f}%, C = {hc['all']['C']:.4f}, "
            f"A_n = {na['all']:.4f} "
            f"-> Agent captures **{na['all']*100:.1f}%** of human test-retest consistency. "
            f"Agent MAE = {am['all']['MAE']:.4f} (Human baseline MAE = {hm['all']['MAE']:.4f})."
        )

    lines += [
        "",
        "### MAE Interpretation (Section 4.8.4)",
        "- Lower agent MAE relative to human baseline MAE indicates the agent produces responses as",
        "  close to Wave-2 ground truth as the participant's own Wave-1 responses did.",
        "- On **BFI Likert-scale** items (1-5 scale), agent MAE reflects personality trait prediction accuracy.",
        "- On **GSS categorical** items, MAE reflects ordinal proximity of predicted social attitudes.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 72)
    print("  SECTION 4.8 EVALUATION: C / A / A_n / MAE (Per Participant)")
    print("=" * 72)

    all_results = []

    for username in PARTICIPANTS:
        print(f"\n  Processing: {username}")
        result = evaluate_participant(username)
        if result is None:
            continue
        all_results.append(result)

        hc = result["section_481_human_consistency"]
        aa = result["section_482_agent_agreement"]
        na = result["section_483_normalized_accuracy"]
        am = result["section_484_agent_mae"]
        hm = result["section_484_human_mae_baseline"]

        print(f"    4.8.1 Human C   : all={hc['all']['C']:.4f}  gss={hc['gss']['C']:.4f}  bfi={hc['bfi']['C']:.4f}")
        print(f"    4.8.2 Agent A%  : all={aa['all']['A']:.2f}%  gss={aa['gss']['A']:.2f}%  bfi={aa['bfi']['A']:.2f}%")
        print(f"    4.8.3 A_n       : all={na['all']:.4f}  gss={na['gss']:.4f}  bfi={na['bfi']:.4f}")
        print(f"    4.8.4 Agent MAE : all={am['all']['MAE']:.4f}  gss={am['gss']['MAE']:.4f}  bfi={am['bfi']['MAE']:.4f}")
        print(f"    4.8.4 Human MAE : all={hm['all']['MAE']:.4f}  gss={hm['gss']['MAE']:.4f}  bfi={hm['bfi']['MAE']:.4f}")

    if not all_results:
        print("No participants evaluated -- check paths and data.")
        return

    # Save JSON
    json_path = OUTPUT_DIR / "section_48_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  [JSON] Saved: {json_path}")

    # Save CSV
    csv_path = OUTPUT_DIR / "section_48_metrics.csv"
    rows = [generate_csv_row(p) for p in all_results]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] Saved: {csv_path}")

    # Save Markdown report
    md_path = OUTPUT_DIR / "section_48_metrics_report.md"
    generate_markdown_report(all_results, md_path)
    print(f"  [MD]  Saved: {md_path}")

    # Print cohort summary
    print("\n" + "=" * 72)
    print("  COHORT SUMMARY (All Participants)")
    print("=" * 72)
    means = {
        "C_all": [p["section_481_human_consistency"]["all"]["C"] for p in all_results],
        "A_all": [p["section_482_agent_agreement"]["all"]["A"] for p in all_results],
        "An_all": [p["section_483_normalized_accuracy"]["all"] for p in all_results],
        "Agent_MAE": [p["section_484_agent_mae"]["all"]["MAE"] for p in all_results],
        "Human_MAE": [p["section_484_human_mae_baseline"]["all"]["MAE"] for p in all_results],
    }
    for k, vs in means.items():
        valid = [v for v in vs if v is not None]
        print(f"  Mean {k:20s}: {sum(valid)/len(valid):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

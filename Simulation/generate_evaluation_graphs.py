#!/usr/bin/env python3
"""
generate_evaluation_graphs.py -- Generates 5 overall cohort-level evaluation graphs
and stores them in export/evaluation_scores/.

Graphs created:
  1. bfi_trait_correlation_graph.png      (Trait Pearson r: Human vs Agent + % retention)
  2. consistency_and_accuracy_barplot.png  (Human Consistency C vs Agent Agreement A%)
  3. normalized_accuracy_barplot.png       (Normalized Accuracy A_n vs Human Parity 1.0)
  4. mae_comparison_barplot.png            (Agent MAE vs Human Baseline MAE across domains)
  5. bfi_fisher_z_retention_graph.png      (Fisher z-transform values & % ceiling captured)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Style setup for professional research publication figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "font.family": "sans-serif",
})

BASE_DIR = Path(r"c:\Users\mayil\Music\Data_Processor")
SCORES_DIR = BASE_DIR / "export" / "evaluation_scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with open(SCORES_DIR / "bfi_pearson_fisher_z.json", "r", encoding="utf-8") as f:
        bfi_data = json.load(f)
    metric_file = SCORES_DIR / "metrics.json" if (SCORES_DIR / "metrics.json").exists() else SCORES_DIR / "section_48_metrics.json"
    with open(metric_file, "r", encoding="utf-8") as f:
        s48_data = json.load(f)
    return bfi_data, s48_data


def compute_cohort_stats(bfi_data, s48_data):
    # 1. BFI trait stats
    traits = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    by_p = bfi_data["by_participant"]

    human_trait_r = []
    agent_trait_r = []
    human_trait_z = []
    agent_trait_z = []
    retention_pct = []

    for t in traits:
        hz_vals = [p["comparisons"]["human_wave1_vs_wave2"]["traits"][t]["fisher_z"] for p in by_p.values()]
        az_vals = [p["comparisons"]["agent_both_vs_wave2"]["traits"][t]["fisher_z"] for p in by_p.values()]
        mean_hz = sum(hz_vals) / len(hz_vals)
        mean_az = sum(az_vals) / len(az_vals)
        mean_hr = math.tanh(mean_hz)
        mean_ar = math.tanh(mean_az)

        human_trait_z.append(mean_hz)
        agent_trait_z.append(mean_az)
        human_trait_r.append(mean_hr)
        agent_trait_r.append(mean_ar)
        retention = (mean_ar / mean_hr * 100.0) if mean_hr > 0 else 0.0
        retention_pct.append(retention)

    # Overall BFI
    mean_ov_hz = sum(p["comparisons"]["human_wave1_vs_wave2"]["average_fisher_z"] for p in by_p.values()) / len(by_p)
    mean_ov_az = sum(p["comparisons"]["agent_both_vs_wave2"]["average_fisher_z"] for p in by_p.values()) / len(by_p)
    mean_ov_hr = math.tanh(mean_ov_hz)
    mean_ov_ar = math.tanh(mean_ov_az)
    ov_retention = (mean_ov_ar / mean_ov_hr) * 100.0

    # 2. Section 48 cohort stats
    domains = ["all", "gss", "bfi"]
    human_c = {d: np.mean([p["section_481_human_consistency"][d]["C"] * 100 for p in s48_data]) for d in domains}
    agent_a = {d: np.mean([p["section_482_agent_agreement"][d]["A"] for p in s48_data]) for d in domains}
    norm_acc = {d: np.mean([p["section_483_normalized_accuracy"][d] for p in s48_data]) for d in domains}
    agent_mae = {d: np.mean([p["section_484_agent_mae"][d]["MAE"] for p in s48_data]) for d in domains}
    human_mae = {d: np.mean([p["section_484_human_mae_baseline"][d]["MAE"] for p in s48_data]) for d in domains}

    return {
        "traits": traits,
        "human_trait_r": human_trait_r,
        "agent_trait_r": agent_trait_r,
        "human_trait_z": human_trait_z,
        "agent_trait_z": agent_trait_z,
        "retention_pct": retention_pct,
        "mean_ov_hr": mean_ov_hr,
        "mean_ov_ar": mean_ov_ar,
        "mean_ov_hz": mean_ov_hz,
        "mean_ov_az": mean_ov_az,
        "ov_retention": ov_retention,
        "human_c": human_c,
        "agent_a": agent_a,
        "norm_acc": norm_acc,
        "agent_mae": agent_mae,
        "human_mae": human_mae,
    }


def plot_graph_1(stats):
    """Graph 1: Trait Correlation Graph (Pearson r: Human vs Agent + % retention)"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = stats["traits"] + ["OVERALL"]
    h_vals = stats["human_trait_r"] + [stats["mean_ov_hr"]]
    a_vals = stats["agent_trait_r"] + [stats["mean_ov_ar"]]
    ret_vals = stats["retention_pct"] + [stats["ov_retention"]]

    x = np.arange(len(categories))
    width = 0.36

    rects1 = ax.bar(x - width/2, h_vals, width, label="Human Test-Retest (W1 vs W2)", color="#2b5c8f", alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, a_vals, width, label="Agent Replication (vs W2)", color="#e07a5f", alpha=0.95, edgecolor="black", linewidth=0.8)

    # Annotate retention rate %
    for i in range(len(categories)):
        top = max(h_vals[i], a_vals[i])
        ax.text(x[i] + width/2, a_vals[i] + 0.02, f"{a_vals[i]:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#a33d26")
        ax.text(x[i] - width/2, h_vals[i] + 0.02, f"{h_vals[i]:.2f}", ha="center", va="bottom", fontsize=10, color="#1a3d60")
        ax.text(x[i], top + 0.09, f"[{ret_vals[i]:.1f}% ceiling]", ha="center", va="bottom", fontsize=9.5, fontweight="semibold", color="#264653")

    ax.set_ylabel("Pearson Correlation (r)", fontweight="bold")
    ax.set_title("Graph 1: Big Five Personality Trait Correlation (Human vs. Agent)", pad=20, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="semibold")
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    out = SCORES_DIR / "bfi_trait_correlation_graph.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_graph_2(stats):
    """Graph 2: Bar Plot for Consistency & Accuracy (Human Consistency C vs Agent Agreement A%)"""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    labels = ["All Questions\n(84 Items)", "GSS Categorical\n(40 Items)", "BFI Likert Scale\n(44 Items)"]
    c_vals = [stats["human_c"]["all"], stats["human_c"]["gss"], stats["human_c"]["bfi"]]
    a_vals = [stats["agent_a"]["all"], stats["agent_a"]["gss"], stats["agent_a"]["bfi"]]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, c_vals, width, label="Human Consistency (C %)", color="#3d5a80", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, a_vals, width, label="Agent Exact Match (A %)", color="#ee6c4d", edgecolor="black", linewidth=0.8)

    for i in range(len(labels)):
        ax.text(x[i] - width/2, c_vals[i] + 1.2, f"{c_vals[i]:.1f}%", ha="center", va="bottom", fontweight="bold", color="#293241")
        ax.text(x[i] + width/2, a_vals[i] + 1.2, f"{a_vals[i]:.1f}%", ha="center", va="bottom", fontweight="bold", color="#d9532f")

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Graph 2: Human Consistency (C) vs. Agent Agreement Rate (A)", pad=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="semibold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    out = SCORES_DIR / "consistency_and_accuracy_barplot.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_graph_3(stats):
    """Graph 3: Bar Plot for Normalized Accuracy (A_n) across domains with parity threshold line"""
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    domains = ["All Questions", "GSS Social Attitudes", "BFI Personality"]
    vals = [stats["norm_acc"]["all"], stats["norm_acc"]["gss"], stats["norm_acc"]["bfi"]]
    pcts = [v * 100 for v in vals]

    x = np.arange(len(domains))
    colors = ["#2a9d8f", "#457b9d", "#e76f51"]

    bars = ax.bar(x, vals, width=0.45, color=colors, edgecolor="black", linewidth=0.9)

    # Parity baseline at 1.0
    ax.axhline(1.0, color="#e63946", linestyle="--", linewidth=1.8, label="Human Parity Ceiling (1.0 = 100% human stability)")

    for bar, val, pct in zip(bars, vals, pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{val:.4f}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Normalized Accuracy (A_n = A / 100·C)", fontweight="bold")
    ax.set_title("Graph 3: Normalized Accuracy across Evaluation Domains", pad=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontweight="semibold")
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    out = SCORES_DIR / "normalized_accuracy_barplot.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_graph_4(stats):
    """Graph 4: Mean Absolute Error (MAE) Comparison (Agent vs Human Baseline)"""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    labels = ["All Questions\n(Ordinal Scale)", "GSS Categorical\n(Attitude Scale)", "BFI Personality\n(1-5 Likert Scale)"]
    ag_mae = [stats["agent_mae"]["all"], stats["agent_mae"]["gss"], stats["agent_mae"]["bfi"]]
    hu_mae = [stats["human_mae"]["all"], stats["human_mae"]["gss"], stats["human_mae"]["bfi"]]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, ag_mae, width, label="Agent MAE (Agent vs W2)", color="#e76f51", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, hu_mae, width, label="Human Baseline MAE (W1 vs W2)", color="#264653", edgecolor="black", linewidth=0.8)

    for i in range(len(labels)):
        ax.text(x[i] - width/2, ag_mae[i] + 0.02, f"{ag_mae[i]:.2f}", ha="center", va="bottom", fontweight="bold", color="#b5371c")
        ax.text(x[i] + width/2, hu_mae[i] + 0.02, f"{hu_mae[i]:.2f}", ha="center", va="bottom", fontweight="bold", color="#192e37")

    # Add 1-point threshold line
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="1.0 Scale Unit Deviation")

    ax.set_ylabel("Mean Absolute Error (Scale Units)", fontweight="bold")
    ax.set_title("Graph 4: Mean Absolute Error (MAE): Agent vs. Human Baseline", pad=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="semibold")
    ax.set_ylim(0, 1.6)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    out = SCORES_DIR / "mae_comparison_barplot.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_graph_5(stats):
    """Graph 5: Fisher z-Transform & Ceiling Retention Rate"""
    fig, ax1 = plt.subplots(figsize=(10.5, 6), dpi=300)

    categories = stats["traits"] + ["OVERALL"]
    hz_vals = stats["human_trait_z"] + [stats["mean_ov_hz"]]
    az_vals = stats["agent_trait_z"] + [stats["mean_ov_az"]]
    ret_vals = stats["retention_pct"] + [stats["ov_retention"]]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, hz_vals, width, label="Human Baseline Fisher z", color="#1d3557", edgecolor="black", linewidth=0.8)
    rects2 = ax1.bar(x + width/2, az_vals, width, label="Agent Fisher z", color="#457b9d", edgecolor="black", linewidth=0.8)

    for i in range(len(categories)):
        ax1.text(x[i] - width/2, hz_vals[i] + 0.02, f"{hz_vals[i]:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1d3557")
        ax1.text(x[i] + width/2, az_vals[i] + 0.02, f"{az_vals[i]:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2a536b")

    # Line overlay on second axis for % retention
    ax2 = ax1.twinx()
    ax2.plot(x, ret_vals, color="#e63946", marker="o", linewidth=2.5, markersize=8, label="% Human Ceiling Captured")
    ax2.set_ylabel("% Human Ceiling Captured", color="#e63946", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#e63946")
    ax2.set_ylim(0, 120)
    ax2.grid(False)  # avoid overlapping grids

    for i in range(len(categories)):
        ax2.text(x[i], ret_vals[i] + 4, f"{ret_vals[i]:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#e63946")

    ax1.set_ylabel("Fisher z Value (arctanh(r))", fontweight="bold")
    ax1.set_title("Graph 5: Fisher z-Transformation & Human Consistency Retention Rate", pad=20, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontweight="semibold")
    ax1.set_ylim(-0.1, 1.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    plt.tight_layout()

    out = SCORES_DIR / "bfi_fisher_z_retention_graph.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("=" * 60)
    print("  GENERATING 5 EVALUATION GRAPHS (COHORT-LEVEL)")
    print("=" * 60)

    bfi_data, s48_data = load_data()
    stats = compute_cohort_stats(bfi_data, s48_data)

    plot_graph_1(stats)
    plot_graph_2(stats)
    plot_graph_3(stats)
    plot_graph_4(stats)
    plot_graph_5(stats)

    print("\nAll 5 graphs successfully saved to:")
    print(f"  {SCORES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

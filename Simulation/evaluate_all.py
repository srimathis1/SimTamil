#!/usr/bin/env python3
"""
evaluate_all.py -- all evaluation metrics + plots for one participant.

    python evaluate_all.py Mounees
    python evaluate_all.py Mounees --conditions interview both --n-boot 5000

It recomputes the agent-vs-Wave-2 comparison in memory (no stale files), so
just run agent_surv.py (for any conditions) and then this script.

Writes to <participant>/agent_eval/ :
    metrics.json        every number below, per condition
    summary.csv         one row per condition (paste into your report)
    item_level.csv      every scored item: Wave 1, Wave 2, each agent, match or not
    report.md           readable summary + caveats
    plots/*.png         6 figures

Metrics (per condition; "human" = the participant's own Wave-1 answers used to
predict Wave 2, i.e. the test-retest ceiling):
  GSS   exact accuracy (+95% bootstrap CI over items), chance level,
        chance-corrected accuracy, normalized accuracy = agent / human,
        ordinal scoring on ordered items (mean normalised distance, within 1),
        accuracy by section.
  BFI   exact accuracy, within-1 accuracy, item MAE, within-person item
        correlation, normalized accuracy / correlation, trait scores
        (reverse-keyed items corrected), trait MAE and signed bias,
        answer distribution, extreme/neutral answer rate and spread
        (to see whether the agent flattens answers toward the middle).
  All   pooled exact accuracy and normalized accuracy with CI.

With one participant, all correlations are within-person across items and all
CIs come from resampling items; neither is a population-level estimate.
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import agent_vs_wave2 as cmp  # noqa: E402
import twin_common as tc  # noqa: E402

BASE_LABELS = {"human_wave1": "Human: Wave 1 vs Wave 2", "demographics": "Demographics only",
               "interview": "Interview", "survey": "Survey", "both": "Interview + survey"}
BASE_COLORS = {"human_wave1": "#8c8c8c", "demographics": "#bcbd22", "interview": "#1f77b4",
               "survey": "#2ca02c", "both": "#9467bd"}
EXTRA_COLORS = ["#e377c2", "#17becf", "#ff7f0e", "#8c564b"]
PUBLISHED_BAND = (0.82, 0.88)   # Park et al. 2024 (GSS 0.82-0.86) and Twin-2K-500 (0.88), U.S. samples
TRAITS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 150})


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------
def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def pearson(x, y):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pct_ci(a):
    return [r4(np.percentile(a, 2.5)), r4(np.percentile(a, 97.5))]


def boot(agent_flags, human_flags, n_boot, rng):
    """95% bootstrap CI (resampling items) for agent accuracy and the agent/human ratio."""
    a, h = np.asarray(agent_flags, float), np.asarray(human_flags, float)
    n = len(a)
    if n == 0:
        return {"agent": None, "ratio": None}
    idx = rng.integers(0, n, size=(n_boot, n))
    am, hm = a[idx].mean(1), h[idx].mean(1)
    ok = hm > 0
    ratio = am[ok] / hm[ok]
    return {"agent": pct_ci(am), "ratio": pct_ci(ratio) if ratio.size > 20 else None}


def rc(value, reverse):
    return 6 - value if reverse else value


# --------------------------------------------------------------------------
# metrics for one condition
# --------------------------------------------------------------------------
def condition_metrics(cond, res, human_res, n_boot, rng):
    recs, hum = res["records"], human_res["records"]
    is_human = cond == tc.HUMAN_CONDITION
    m = {"condition": cond, "leakage_risk": bool(res["meta"].get("leakage_risk")),
         "meta": res["meta"], "n_scored": len(recs), "n_failed": sum(r["failed"] for r in recs)}

    gss = [(r, h) for r, h in zip(recs, hum) if r["kind"] == "gss"]
    bfi = [(r, h) for r, h in zip(recs, hum) if r["kind"] == "bfi"]
    m["n_gss"], m["n_bfi"] = len(gss), len(bfi)

    # ---- GSS ----------------------------------------------------------------
    g = {}
    if gss:
        a, h = [r["exact"] for r, _ in gss], [x["exact"] for _, x in gss]
        acc, hacc, chance = mean(a), mean(h), mean(1 / r["n_options"] for r, _ in gss)
        b = boot(a, h, n_boot, rng)
        g.update(exact=r4(acc), exact_ci=b["agent"], human_exact=r4(hacc), chance=r4(chance),
                 chance_corrected=r4((acc - chance) / (1 - chance)),
                 normalized=None if is_human or not hacc else r4(acc / hacc),
                 normalized_ci=None if is_human else b["ratio"])
        ordp = [(r, x) for r, x in gss if r["ordinal_distance"] is not None and x["ordinal_distance"] is not None]
        if ordp:
            nd = lambda rec: rec["ordinal_distance"] / (rec["n_options"] - 1)  # noqa: E731
            g["ordinal"] = {"n_items": len(ordp),
                            "mean_norm_distance": r4(mean(nd(r) for r, _ in ordp)),
                            "human_mean_norm_distance": r4(mean(nd(x) for _, x in ordp)),
                            "within1": r4(mean(r["ordinal_distance"] <= 1 for r, _ in ordp)),
                            "human_within1": r4(mean(x["ordinal_distance"] <= 1 for _, x in ordp))}
        secs = OrderedDict()
        for r, x in gss:
            s = secs.setdefault(r["section"], {"a": [], "h": []})
            s["a"].append(r["exact"]); s["h"].append(x["exact"])
        g["by_section"] = {k: {"n": len(v["a"]), "exact": r4(mean(v["a"])), "human_exact": r4(mean(v["h"]))}
                           for k, v in secs.items()}
    m["gss"] = g

    # ---- BFI ----------------------------------------------------------------
    f = {}
    if bfi:
        a, h = [r["exact"] for r, _ in bfi], [x["exact"] for _, x in bfi]
        w1 = [bool(r["within1"]) for r, _ in bfi]
        hw1 = [bool(x["within1"]) for _, x in bfi]
        acc, hacc = mean(a), mean(h)
        b = boot(a, h, n_boot, rng)
        ok = [(r, x) for r, x in bfi if not r["failed"]]
        item_mae = mean(abs(r["pred"] - r["wave2"]) for r, _ in ok)
        h_item_mae = mean(abs(x["pred"] - x["wave2"]) for _, x in ok)
        r_agent = pearson([r["pred"] for r, _ in ok], [r["wave2"] for r, _ in ok])
        r_human = pearson([x["pred"] for _, x in ok], [x["wave2"] for _, x in ok])
        f.update(exact=r4(acc), exact_ci=b["agent"], human_exact=r4(hacc), chance_exact=0.2,
                 within1=r4(mean(w1)), human_within1=r4(mean(hw1)),
                 normalized_exact=None if is_human or not hacc else r4(acc / hacc),
                 normalized_exact_ci=None if is_human else b["ratio"],
                 normalized_within1=None if is_human or not mean(hw1) else r4(mean(w1) / mean(hw1)),
                 item_mae=r4(item_mae), human_item_mae=r4(h_item_mae),
                 item_pearson_r=r4(r_agent), human_item_pearson_r=r4(r_human),
                 normalized_r=None if is_human or not r_agent or not r_human or r_human <= 0 else r4(r_agent / r_human))
        # trait scores (reverse-keyed items corrected)
        traits = OrderedDict()
        for r, x in ok:
            t = traits.setdefault(r["trait"], {"w1": [], "w2": [], "ag": []})
            t["w1"].append(rc(r["wave1"], r["reverse"])); t["w2"].append(rc(r["wave2"], r["reverse"]))
            t["ag"].append(rc(r["pred"], r["reverse"]))
        f["traits"] = {}
        for name, t in traits.items():
            w1m, w2m, agm = mean(t["w1"]), mean(t["w2"]), mean(t["ag"])
            f["traits"][name] = {"n_items": len(t["w2"]), "wave1_mean": r4(w1m), "wave2_mean": r4(w2m),
                                 "agent_mean": r4(agm), "abs_error": r4(abs(agm - w2m)),
                                 "signed_bias": r4(agm - w2m), "human_abs_error": r4(abs(w1m - w2m))}
        f["trait_mae"] = r4(mean(v["abs_error"] for v in f["traits"].values()))
        f["human_trait_mae"] = r4(mean(v["human_abs_error"] for v in f["traits"].values()))
        by_trait = OrderedDict()
        for r, x in bfi:
            s = by_trait.setdefault(r["trait"], {"a": [], "h": []})
            s["a"].append(r["exact"]); s["h"].append(x["exact"])
        f["by_trait"] = {k: {"n": len(v["a"]), "exact": r4(mean(v["a"])), "human_exact": r4(mean(v["h"]))}
                         for k, v in by_trait.items()}
        # answer distribution: does the agent flatten answers toward the middle?
        def dist(vals):
            vals = list(vals)
            return {"proportions": [r4(sum(v == k for v in vals) / len(vals)) for k in range(1, 6)] if vals else None,
                    "extreme_rate": r4(mean(v in (1, 5) for v in vals)), "neutral_rate": r4(mean(v == 3 for v in vals)),
                    "std": r4(float(np.std(vals))) if vals else None}
        f["distribution"] = {"wave2": dist(r["wave2"] for r, _ in ok), "wave1": dist(r["wave1"] for r, _ in ok),
                             "agent": dist(r["pred"] for r, _ in ok)}
    m["bfi"] = f

    # ---- pooled -------------------------------------------------------------
    a, h = [r["exact"] for r in recs], [x["exact"] for x in hum]
    chance_all = mean([1 / r["n_options"] for r, _ in gss] + [0.2] * len(bfi))
    b = boot(a, h, n_boot, rng)
    acc, hacc = mean(a), mean(h)
    m["overall"] = {"exact": r4(acc), "exact_ci": b["agent"], "human_exact": r4(hacc), "chance": r4(chance_all),
                    "normalized": None if is_human or not hacc else r4(acc / hacc),
                    "normalized_ci": None if is_human else b["ratio"]}
    return m


# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------
def order_conditions(names):
    first = [tc.HUMAN_CONDITION] if tc.HUMAN_CONDITION in names else []
    known = [n for n in ("demographics", "interview", "survey", "both") if n in names]
    rest = sorted(n for n in names if n not in first + known and n.replace("_noref", "") not in BASE_LABELS)
    noref = sorted(n for n in names if n not in first + known and n.replace("_noref", "") in BASE_LABELS)
    return first + known + noref + rest


def style(cond, metrics):
    base = cond.replace("_noref", "")
    label = BASE_LABELS.get(base, cond)
    if cond.endswith("_noref"):
        label += " (no reflections)"
    if metrics[cond]["leakage_risk"]:
        label += " *"
    color = BASE_COLORS.get(base)
    if color is None:
        color = EXTRA_COLORS[sum(map(ord, base)) % len(EXTRA_COLORS)]
    return label, color, ("//" if cond.endswith("_noref") else None)


def footnote(fig, metrics):
    if any(m["leakage_risk"] for m in metrics.values()):
        fig.text(0.01, 0.005, "* reflections were generated from the full Wave-1 survey and may reveal the hidden "
                 "answers: treat as an upper bound, not a prediction.", fontsize=7, color="#555", va="bottom")


def grouped_bars(ax, categories, series, styles, errs=None, fmt="{:.0%}", labels=True):
    n = len(series)
    w = 0.82 / max(n, 1)
    for i, vals in enumerate(series):
        label, color, hatch = styles[i]
        x = np.arange(len(categories)) + (i - (n - 1) / 2) * w
        h = np.array([np.nan if v is None else v for v in vals], float)
        ax.bar(x, h, w * 0.92, label=label, color=color, hatch=hatch,
               edgecolor="black" if hatch else "none", linewidth=0.4)
        if errs and errs[i]:
            lo = np.array([np.nan if (e is None or v is None) else v - e[0] for v, e in zip(vals, errs[i])], float)
            hi = np.array([np.nan if (e is None or v is None) else e[1] - v for v, e in zip(vals, errs[i])], float)
            ax.errorbar(x, h, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], fmt="none",
                        ecolor="black", capsize=2, lw=0.8)
        if labels:
            for xi, hi_ in zip(x, h):
                if not np.isnan(hi_):
                    ax.text(xi, hi_ + 0.012, fmt.format(hi_), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_overall(metrics, conds, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    groups = ["GSS", "BFI", "All items"]
    keys = [("gss", "exact", "exact_ci"), ("bfi", "exact", "exact_ci"), ("overall", "exact", "exact_ci")]
    st = [style(c, metrics) for c in conds]
    series = [[metrics[c][k][v] if metrics[c][k] else None for k, v, _ in keys] for c in conds]
    errs = [[(metrics[c][k][e] if (c != tc.HUMAN_CONDITION and metrics[c][k]) else None) for k, _, e in keys] for c in conds]
    ax = axes[0]
    grouped_bars(ax, groups, series, st, errs)
    ref = metrics[conds[0]]
    chance = [ref["gss"].get("chance"), 0.2, ref["overall"].get("chance")]
    for xi, c in enumerate(chance):
        if c is not None:
            ax.hlines(c, xi - 0.43, xi + 0.43, colors="black", linestyles=":", lw=1.2)
    ax.plot([], [], "k:", label="chance")
    ax.set_ylim(0, 1.12); ax.set_ylabel("exact-match accuracy vs Wave 2")
    ax.set_title("Raw accuracy (95% CI over items)")
    ax.legend(fontsize=7, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))

    agents = [c for c in conds if c != tc.HUMAN_CONDITION]
    ax = axes[1]
    nk = [("gss", "normalized", "normalized_ci"), ("bfi", "normalized_exact", "normalized_exact_ci"),
          ("overall", "normalized", "normalized_ci")]
    series = [[metrics[c][k][v] if metrics[c][k] else None for k, v, _ in nk] for c in agents]
    errs = [[(metrics[c][k][e] if metrics[c][k] else None) for k, _, e in nk] for c in agents]
    ax.axhspan(*PUBLISHED_BAND, color="#cccccc", alpha=0.5, lw=0)
    ax.text(2.45, sum(PUBLISHED_BAND) / 2, "published U.S.\nrange 0.82-0.88", fontsize=7, va="center", ha="right", color="#444")
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.text(-0.45, 1.01, "human retest = 1.0", fontsize=7, va="bottom")
    grouped_bars(ax, groups, series, [style(c, metrics) for c in agents], errs, fmt="{:.2f}")
    top = max([v for s in series for v in s if v is not None] + [1.0])
    ax.set_ylim(0, max(1.25, top + 0.15)); ax.set_ylabel("normalized accuracy = agent / human")
    ax.set_title("Normalized accuracy (95% CI over items)")
    fig.suptitle(title, fontsize=11, y=1.0)
    footnote(fig, metrics)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_by_section(metrics, conds, title, path):
    ref = metrics[tc.HUMAN_CONDITION]
    cats = [(f"GSS {k}", ("gss", "by_section", k)) for k in ref["gss"].get("by_section", {})]
    cats += [(f"BFI {k}", ("bfi", "by_trait", k)) for k in ref["bfi"].get("by_trait", {})]
    if not cats:
        return
    labels = [f"{textwrap.fill(n, 16)}\n(n={ref[a][b][k]['n']})" for n, (a, b, k) in cats]
    series = [[metrics[c][a][b].get(k, {}).get("exact") if metrics[c][a].get(b) else None for _, (a, b, k) in cats] for c in conds]
    fig, ax = plt.subplots(figsize=(max(9, 1.35 * len(cats)), 5.2))
    grouped_bars(ax, labels, series, [style(c, metrics) for c in conds], labels=False)
    ax.set_ylim(0, 1.05); ax.set_ylabel("exact-match accuracy vs Wave 2")
    ax.set_title(title); ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    footnote(fig, metrics)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_bfi_traits(metrics, conds, title, path):
    ref = metrics[tc.HUMAN_CONDITION]["bfi"].get("traits")
    if not ref:
        return
    traits = [t for t in TRAITS if t in ref] + [t for t in ref if t not in TRAITS]
    agents = [c for c in conds if c != tc.HUMAN_CONDITION]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    series = [[ref[t]["wave1_mean"] for t in traits], [ref[t]["wave2_mean"] for t in traits]]
    styles = [("Human Wave 1", "#cfcfcf", None), ("Human Wave 2 (truth)", "#333333", None)]
    for c in agents:
        series.append([metrics[c]["bfi"]["traits"][t]["agent_mean"] if t in metrics[c]["bfi"].get("traits", {}) else None for t in traits])
        styles.append(style(c, metrics))
    grouped_bars(ax, traits, series, styles, fmt="{:.1f}")
    ax.set_ylim(1, 5.4); ax.set_ylabel("trait score (1-5, reverse-keyed items corrected)")
    ax.set_title("Trait scores")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax = axes[1]
    series = [[ref[t]["wave1_mean"] - ref[t]["wave2_mean"] for t in traits]]
    styles = [("Human Wave 1 - Wave 2", "#8c8c8c", None)]
    for c in agents:
        tr = metrics[c]["bfi"].get("traits", {})
        series.append([tr[t]["signed_bias"] if t in tr else None for t in traits])
        styles.append(style(c, metrics))
    grouped_bars(ax, traits, series, styles, fmt="{:+.1f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("signed bias = predicted - Wave 2"); ax.set_title("Direction of the error (0 = unbiased)")
    fig.suptitle(title, fontsize=11)
    footnote(fig, metrics)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_bfi_scatter(results, metrics, conds, title, path, rng):
    ncol = min(3, len(conds)); nrow = -(-len(conds) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.2 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), conds):
        ax.axis("on")
        rs = [r for r in results[c]["records"] if r["kind"] == "bfi" and not r["failed"]]
        if not rs:
            ax.set_title(style(c, metrics)[0] + "\n(no BFI answers)", fontsize=8)
            continue
        x = np.array([r["wave2"] for r in rs], float) + rng.uniform(-0.13, 0.13, len(rs))
        y = np.array([r["pred"] for r in rs], float) + rng.uniform(-0.13, 0.13, len(rs))
        label, color, _ = style(c, metrics)
        ax.plot([0.5, 5.5], [0.5, 5.5], color="#999", lw=0.8)
        ax.scatter(x, y, s=22, color=color, alpha=0.75, edgecolor="none")
        rr = metrics[c]["bfi"].get("item_pearson_r")
        ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5); ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
        ax.set_xlabel("Wave 2 answer (truth)"); ax.set_ylabel("Wave 1 answer" if c == tc.HUMAN_CONDITION else "agent answer")
        ax.set_title(f"{label}\nexact {metrics[c]['bfi']['exact']:.0%}   r = {rr if rr is not None else 'n/a'}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    footnote(fig, metrics)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_distribution(metrics, conds, title, path):
    ref = metrics[tc.HUMAN_CONDITION]["bfi"].get("distribution")
    if not ref:
        return
    cats = ["1\nstrongly\ndisagree", "2", "3\nneutral", "4", "5\nstrongly\nagree"]
    series = [ref["wave2"]["proportions"], ref["wave1"]["proportions"]]
    styles = [("Human Wave 2 (truth)", "#333333", None), ("Human Wave 1", "#cfcfcf", None)]
    for c in conds:
        if c == tc.HUMAN_CONDITION:
            continue
        d = metrics[c]["bfi"].get("distribution")
        if d:
            series.append(d["agent"]["proportions"]); styles.append(style(c, metrics))
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped_bars(ax, cats, series, styles, fmt="{:.0%}", labels=False)
    ax.set_ylabel("share of BFI answers"); ax.set_title(title)
    txt = "   ".join(f"{s[0].split(' (')[0]}: extreme {metrics[c]['bfi']['distribution']['agent']['extreme_rate']:.0%}"
                     for s, c in zip(styles[2:], [x for x in conds if x != tc.HUMAN_CONDITION]))
    ref_ext = ref["wave2"]["extreme_rate"]
    ax.text(0.5, -0.30, f"Share of extreme answers (1 or 5) - Wave 2: {ref_ext:.0%}   {txt}", transform=ax.transAxes,
            ha="center", fontsize=7.5)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    footnote(fig, metrics)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_heatmap(results, metrics, conds, title, path):
    keys = [r["key"] for r in results[conds[0]]["records"]]
    mat = np.zeros((len(keys), len(conds)))
    for j, c in enumerate(conds):
        for i, r in enumerate(results[c]["records"]):
            mat[i, j] = 2 if r["failed"] else (1 if r["exact"] else 0)
    fig, ax = plt.subplots(figsize=(1.2 + 1.3 * len(conds), 1.5 + 0.17 * len(keys)))
    ax.imshow(mat, aspect="auto", cmap=ListedColormap(["#d62728", "#2ca02c", "#bdbdbd"]), vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=6)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([textwrap.fill(style(c, metrics)[0], 14) for c in conds], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=10, pad=34)
    ax.legend(handles=[Patch(color="#2ca02c", label="matches Wave 2"), Patch(color="#d62728", label="differs"),
                       Patch(color="#bdbdbd", label="agent gave no valid answer")],
              fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3)
    footnote(fig, metrics)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------
# tables and report
# --------------------------------------------------------------------------
def fmt_pct(v, ci=None):
    if v is None:
        return "-"
    s = f"{v:.1%}"
    return s + (f" ({ci[0]:.0%}-{ci[1]:.0%})" if ci else "")


def fmt_num(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


def write_summary_csv(metrics, conds, path):
    cols = ["condition", "label", "leakage_risk", "n_scored", "n_gss", "n_bfi", "n_failed",
            "gss_exact", "gss_exact_ci_lo", "gss_exact_ci_hi", "gss_chance", "gss_chance_corrected",
            "gss_normalized", "gss_norm_ci_lo", "gss_norm_ci_hi", "gss_ordinal_within1", "gss_ordinal_mean_norm_distance",
            "bfi_exact", "bfi_exact_ci_lo", "bfi_exact_ci_hi", "bfi_within1", "bfi_normalized_exact",
            "bfi_item_mae", "bfi_item_pearson_r", "bfi_normalized_r", "bfi_trait_mae",
            "bfi_extreme_rate", "bfi_neutral_rate", "bfi_answer_std",
            "overall_exact", "overall_norm", "overall_norm_ci_lo", "overall_norm_ci_hi", "human_gss_exact", "human_bfi_exact"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c in conds:
            m = metrics[c]; g, f, o = m["gss"], m["bfi"], m["overall"]
            ci = lambda d, k: (d.get(k) or [None, None])  # noqa: E731
            dist = (f.get("distribution") or {}).get("agent") or {}
            w.writerow([c, style(c, metrics)[0], m["leakage_risk"], m["n_scored"], m["n_gss"], m["n_bfi"], m["n_failed"],
                        g.get("exact"), ci(g, "exact_ci")[0], ci(g, "exact_ci")[1], g.get("chance"), g.get("chance_corrected"),
                        g.get("normalized"), ci(g, "normalized_ci")[0], ci(g, "normalized_ci")[1],
                        (g.get("ordinal") or {}).get("within1"), (g.get("ordinal") or {}).get("mean_norm_distance"),
                        f.get("exact"), ci(f, "exact_ci")[0], ci(f, "exact_ci")[1], f.get("within1"), f.get("normalized_exact"),
                        f.get("item_mae"), f.get("item_pearson_r"), f.get("normalized_r"), f.get("trait_mae"),
                        dist.get("extreme_rate"), dist.get("neutral_rate"), dist.get("std"),
                        o.get("exact"), o.get("normalized"), ci(o, "normalized_ci")[0], ci(o, "normalized_ci")[1],
                        g.get("human_exact"), f.get("human_exact")])


def write_item_csv(results, conds, path):
    base = results[tc.HUMAN_CONDITION]["records"]
    agents = [c for c in conds if c != tc.HUMAN_CONDITION]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "kind", "section", "wave1", "wave2", "human_match"]
                   + [x for c in agents for x in (f"{c}_pred", f"{c}_match")])
        for i, r in enumerate(base):
            row = [r["key"], r["kind"], r["section"], r["wave1_text"], r["wave2_text"], r["exact"]]
            for c in agents:
                rr = results[c]["records"][i]
                row += [rr["pred_text"] if not rr["failed"] else "(no answer)", rr["exact"]]
            w.writerow(row)


def write_report(pdir, metrics, conds, universe, path, n_boot):
    ref = metrics[tc.HUMAN_CONDITION]
    L = [f"# Agent evaluation - {pdir.name}", "",
         f"Scored items: **{ref['n_scored']}** ({ref['n_gss']} GSS, {ref['n_bfi']} BFI). "
         f"Items that are not applicable, or missing in Wave 1 or Wave 2, are excluded "
         f"({len(universe['excluded'])} items; see item_universe.json).", "",
         "## Results", "",
         "| Condition | GSS exact (95% CI) | GSS normalized | BFI exact | BFI within 1 | BFI trait MAE | Overall exact | Overall normalized (95% CI) | No answer |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        m = metrics[c]; g, f, o = m["gss"], m["bfi"], m["overall"]
        L.append(f"| {style(c, metrics)[0]} | {fmt_pct(g.get('exact'), g.get('exact_ci'))} | {fmt_num(g.get('normalized'))} | "
                 f"{fmt_pct(f.get('exact'))} | {fmt_pct(f.get('within1'))} | {fmt_num(f.get('trait_mae'))} | "
                 f"{fmt_pct(o.get('exact'))} | {fmt_num(o.get('normalized'))}"
                 f"{' (' + format(o['normalized_ci'][0], '.2f') + '-' + format(o['normalized_ci'][1], '.2f') + ')' if o.get('normalized_ci') else ''} | {m['n_failed']} |")
    g = ref["gss"]
    L += ["", f"Chance level: GSS {fmt_pct(g.get('chance'))}, BFI 20.0%. The human row is the participant's own "
              "Wave-1 answers used to predict Wave 2 (test-retest ceiling). Published U.S. normalized accuracy: 0.82-0.88 "
              "(Park et al. 2024; Twin-2K-500).", "", "## Read this before quoting the numbers", ""]
    cav = []
    if any(metrics[c]["leakage_risk"] for c in conds):
        cav.append("Conditions marked * use expert reflections generated from the full Wave-1 survey. "
                   "Those reflections can contain the answers that were hidden from the agent, so the result is an upper "
                   "bound (retest-style), not a prediction. Run `agent_surv.py --reflections none` for a clean survey/both result.")
    if ref["n_gss"] < 30 or ref["n_bfi"] < 30:
        cav.append(f"Only {ref['n_gss']} GSS and {ref['n_bfi']} BFI items are scored, so confidence intervals are wide. "
                   "One participant is a pilot, not an estimate for the population.")
    cav.append("Correlations are within-person across items. Park et al. correlate across participants, which needs many participants.")
    cav.append("The GSS/BFI questions are adapted (custom-worded) instruments, so numbers are only roughly comparable with published ones.")
    cav.append("Interview content that overlaps with GSS topics was not removed; the interview agent may partly 'know' those answers.")
    if universe["wave2_unapplicable"]:
        cav.append("Wave 2 contains answers to items that were not applicable to this participant "
                   f"({', '.join(universe['wave2_unapplicable'])}); they are excluded. The Wave-2 form ignores display_condition.")
    if any(metrics[c]["n_failed"] for c in conds):
        cav.append("Some agent answers were missing/invalid; they are counted as wrong.")
    L += [f"- {x}" for x in cav]
    L += ["", "## Metric guide", "",
          "- **Exact accuracy**: share of items where the answer equals the Wave-2 answer. 95% CI = bootstrap over items "
          f"({n_boot} resamples).",
          "- **Normalized accuracy**: agent accuracy / the participant's own Wave-1-vs-Wave-2 accuracy on the same items.",
          "- **Within 1**: answer differs from Wave 2 by at most one scale step (BFI; ordered GSS items).",
          "- **Trait MAE**: mean absolute difference between agent and Wave-2 trait scores (reverse-keyed items corrected).",
          "- **Signed bias**: predicted minus Wave 2; positive = the agent over-states the trait.",
          "- **Extreme / neutral rate**: share of BFI answers that are 1 or 5 / 3. A much lower extreme rate than Wave 2 means the agent flattens answers."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="All evaluation metrics and plots for one participant.")
    ap.add_argument("username")
    ap.add_argument("--conditions", nargs="+", help="only these agent conditions")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--questions", type=Path, default=tc.DEFAULT_QUESTIONS_FILE)
    ap.add_argument("--participants-dir", type=Path, default=tc.DEFAULT_PARTICIPANTS_DIR)
    args = ap.parse_args(argv)

    pdir = tc.find_participant_dir(args.username, args.participants_dir)
    bank = tc.load_question_bank(args.questions)
    out = cmp.compare_all(pdir, bank, args.conditions, write=True)
    results, universe = out["results"], out["universe"]
    if len(results) == 1:
        raise SystemExit("No agent answer files found in agent_answers/ - run agent_surv.py first.")
    if not universe["scored"]:
        raise SystemExit("No items are answered in both Wave 1 and Wave 2, nothing to score.")

    rng = np.random.default_rng(args.seed)
    conds = order_conditions(list(results))
    metrics = OrderedDict((c, condition_metrics(c, results[c], results[tc.HUMAN_CONDITION], args.n_boot, rng)) for c in conds)

    ev = pdir / tc.EVAL_DIRNAME
    plots = ev / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    tc.write_json(ev / "metrics.json", {"participant_folder": pdir.name, "generated_at": tc.now_iso(),
                                       "n_boot": args.n_boot, "seed": args.seed, "conditions": metrics})
    write_summary_csv(metrics, conds, ev / "summary.csv")
    write_item_csv(results, conds, ev / "item_level.csv")
    write_report(pdir, metrics, conds, universe, ev / "report.md", args.n_boot)

    if not args.no_plots:
        n = metrics[tc.HUMAN_CONDITION]["n_scored"]
        tag = f"{pdir.name} (n = {n} items)"
        jobs = [
            ("01_overall_accuracy.png", lambda p: plot_overall(metrics, conds, f"{tag} - accuracy against Wave 2", p)),
            ("02_accuracy_by_section.png", lambda p: plot_by_section(metrics, conds, f"{tag} - accuracy by GSS section and BFI trait", p)),
            ("03_bfi_traits.png", lambda p: plot_bfi_traits(metrics, conds, f"{tag} - Big Five trait scores", p)),
            ("04_bfi_item_scatter.png", lambda p: plot_bfi_scatter(results, metrics, conds, f"{tag} - BFI item answers vs Wave 2", p, np.random.default_rng(args.seed))),
            ("05_bfi_answer_distribution.png", lambda p: plot_distribution(metrics, conds, f"{tag} - BFI answer distribution", p)),
            ("06_item_agreement_heatmap.png", lambda p: plot_heatmap(results, metrics, conds, f"{tag} - item agreement with Wave 2", p)),
        ]
        for name, fn in jobs:
            try:
                fn(plots / name)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] plot {name} failed: {type(exc).__name__}: {exc}")

    print(f"Participant folder: {pdir.name}   scored items: {metrics[tc.HUMAN_CONDITION]['n_scored']}")
    cmp.print_table({c: results[c] for c in conds})
    print(f"\n{'condition':<18}{'GSS norm.':>11}{'BFI norm.':>11}{'all norm.':>11}{'BFI trait MAE':>15}{'BFI r':>8}")
    for c in conds:
        m = metrics[c]
        print(f"{c:<18}{fmt_num(m['gss'].get('normalized')):>11}{fmt_num(m['bfi'].get('normalized_exact')):>11}"
              f"{fmt_num(m['overall'].get('normalized')):>11}{fmt_num(m['bfi'].get('trait_mae')):>15}{fmt_num(m['bfi'].get('item_pearson_r')):>8}")
    print(f"\nSaved to: {ev}\n  metrics.json  summary.csv  item_level.csv  report.md  plots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
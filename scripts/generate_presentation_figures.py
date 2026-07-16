"""
Generate the three headline figures for the presentation article/slides,
with bootstrap-derived error bars (see bootstrap_presentation_stats.py).
Pure matplotlib, reuses the paper's color scheme for visual consistency.

Usage:
    python scripts/bootstrap_presentation_stats.py   # once, or after data changes
    python scripts/generate_presentation_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "presentation_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = OUT_DIR / "bootstrap_stats.json"

BLUE = "#2563EB"
GREEN = "#059669"
AMBER = "#D97706"
PURPLE = "#9333EA"
GREY = "#6B7280"
DBLUE = "#1E3A5F"

ERRBAR_KW = dict(ecolor="#1F2937", elinewidth=1.3, capsize=4, capthick=1.3)

with open(STATS_PATH) as f:
    STATS = json.load(f)


def fig1_replication():
    s = STATS["fig1"]
    layers = [str(l) for l in s["layers"]]
    ethics, ethics_err = s["ethics"]["point"], s["ethics"]["err"]
    synth, synth_err = s["synthetic"]["point"], s["synthetic"]["err"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(layers))
    w = 0.32
    ax.bar(x - w/2, ethics, width=w, yerr=ethics_err, color=BLUE, label="ETHICS (204 items)", zorder=3, error_kw=ERRBAR_KW)
    ax.bar(x + w/2, synth, width=w, yerr=synth_err, color=GREEN, label="Synthetic bank (160 items, confound-free)", zorder=3, error_kw=ERRBAR_KW)
    ax.axhline(4.0, color=GREY, linestyle="--", lw=1, label="Max possible (4 independent traits)")
    ax.set_xticks(x); ax.set_xticklabels([f"Layer {l}" for l in layers], fontsize=11)
    ax.set_ylabel("Effective Dimensionality", fontsize=11)
    ax.set_title("The collapse replicates on an independent, confound-free dataset",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, 4.6)
    for i, (e, ee, sv, se) in enumerate(zip(ethics, ethics_err, synth, synth_err)):
        ax.text(i - w/2, e + ee + 0.09, f"{e:.2f}", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(i + w/2, sv + se + 0.09, f"{sv:.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Error bars: bootstrap std. dev., 1000 resamples of items.", fontsize=7.5, color=GREY, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_replication.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig2_virtue_axis():
    s = STATS["fig2"]
    layers = [str(l) for l in s["layers"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Left: cosine similarity to the shared collapse direction (PC1)
    ax = axes[0]
    cos, cos_err = s["cosine"]["point"], s["cosine"]["err"]
    bars = ax.bar(layers, cos, yerr=cos_err, color=AMBER, zorder=3, error_kw=ERRBAR_KW)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(0.5, 1.05, color=GREEN, alpha=0.12, zorder=0)
    ax.text(2.35, 0.75, "predicted if\nhypothesis true", fontsize=8, color=GREEN, ha="center", va="center")
    ax.set_ylim(-0.5, 1.05)
    ax.set_ylabel("cosine(virtue_axis, shared axis)", fontsize=10.5)
    ax.set_xlabel("Layer", fontsize=10.5)
    ax.set_title("Generic “good-AI” vector vs.\nthe actual shared collapse direction", fontsize=11, fontweight="bold")
    for b, v, e in zip(bars, cos, cos_err):
        y = v - e - 0.09 if v < 0 else v + e + 0.03
        ax.text(b.get_x()+b.get_width()/2, y, f"{v:.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)

    # Right: effective dimensionality, 4-vector vs +virtue_axis (5-vector)
    ax2 = axes[1]
    ed4, ed4_err = s["ed4"]["point"], s["ed4"]["err"]
    ed5, ed5_err = s["ed5"]["point"], s["ed5"]["err"]
    x = np.arange(len(layers)); w = 0.32
    ax2.bar(x - w/2, ed4, width=w, yerr=ed4_err, color=BLUE, label="4 moral traits", zorder=3, error_kw=ERRBAR_KW)
    ax2.bar(x + w/2, ed5, width=w, yerr=ed5_err, color=AMBER, label="+ virtue_axis (5th vector)", zorder=3, error_kw=ERRBAR_KW)
    ax2.set_xticks(x); ax2.set_xticklabels([f"Layer {l}" for l in layers], fontsize=10.5)
    ax2.set_ylabel("Effective Dimensionality", fontsize=10.5)
    ax2.set_title("Adding the generic vector\nmakes things LESS redundant, not more", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 3.0)
    for i, (a, ae, b, be) in enumerate(zip(ed4, ed4_err, ed5, ed5_err)):
        ax2.text(i - w/2, a + ae + 0.06, f"{a:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax2.text(i + w/2, b + be + 0.06, f"{b:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper center", ncol=1, framealpha=1.0)
    ax2.yaxis.grid(True, alpha=0.3, zorder=0); ax2.set_axisbelow(True)

    fig.suptitle("Testing the “it's just generic RLHF alignment” hypothesis — and rejecting it",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.01, 0.01, "Error bars: bootstrap std. dev., 1000 resamples of items.", fontsize=7.5, color=GREY, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_virtue_axis.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig3_not_noise():
    s = STATS["fig3"]
    traits = s["traits"]
    point = s["point"]
    err = s["err"]
    colors = {"Fairness": GREEN, "Harmlessness": AMBER, "Compassion": PURPLE, "Honesty": GREY}
    label_text = {
        "Fairness": "Discriminates,\nexpected direction",
        "Harmlessness": "Discriminates,\nbackwards",
        "Compassion": "Discriminates,\nbackwards",
        "Honesty": "No significant\nsignal (error bar\ncrosses zero)",
    }
    bar_colors = [colors[t] for t in traits]
    labels = [label_text[t] for t in traits]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars = ax.barh(traits, point, xerr=err, color=bar_colors, zorder=3,
                    error_kw=dict(ecolor="#1F2937", elinewidth=1.3, capsize=4, capthick=1.3))
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(-0.65, 0.65)
    ax.set_xlabel("← discriminates backwards        discriminates as expected →\n(distance from chance, strongest layer per trait)", fontsize=9.5)
    ax.set_title("The four traits are not uniformly collapsed noise —\neach behaves differently on the confound-free bank", fontsize=12, fontweight="bold")
    for b, v, e, lab in zip(bars, point, err, labels):
        edge = v + e if v >= 0 else v - e
        x_text = edge + (0.025 if v >= 0 else -0.025)
        ha = "left" if v >= 0 else "right"
        ax.text(x_text, b.get_y()+b.get_height()/2, lab, va="center", ha=ha, fontsize=8.5)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Error bars: bootstrap std. dev., 1000 resamples (stratified by upheld/violated), n=20+20 per trait.",
              fontsize=7.5, color=GREY, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_not_noise.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig4_salience_replication():
    import pandas as pd
    df = pd.read_csv(OUT_DIR / "salience_replication.csv")
    trait_colors = {"honesty": BLUE, "harmlessness": GREEN, "fairness": AMBER, "compassion": PURPLE}

    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    both_sig = (df.ethics_p < 0.05) & (df.synth_p < 0.05)
    one_sig = ((df.ethics_p < 0.05) ^ (df.synth_p < 0.05))
    neither = ~(df.ethics_p < 0.05) & ~(df.synth_p < 0.05)

    for mask, marker, size, alpha, edge in [
        (both_sig, "o", 190, 1.0, "#1F2937"),
        (one_sig, "o", 130, 0.85, "#9CA3AF"),
        (neither, "o", 90, 0.45, "#D1D5DB"),
    ]:
        sub = df[mask]
        colors = [trait_colors[t] for t in sub["trait"]]
        ax.scatter(sub["ethics_auc"], sub["synth_auc"], c=colors, s=size, alpha=alpha,
                   edgecolors=edge, linewidths=1.3, zorder=4)

    # diagonal (perfect agreement) + chance crosshair
    ax.plot([0.2, 0.85], [0.2, 0.85], linestyle="--", color=GREY, lw=1.2, zorder=2, label="Perfect agreement")
    ax.axhline(0.5, color="#D1D5DB", lw=1, zorder=1)
    ax.axvline(0.5, color="#D1D5DB", lw=1, zorder=1)

    # annotate the honesty sign-flip specifically
    h32 = df[(df.trait == "honesty") & (df.layer == 32)].iloc[0]
    h47 = df[(df.trait == "honesty") & (df.layer == 47)].iloc[0]
    ax.annotate("honesty @32\n(both datasets: real signal)", (h32.ethics_auc, h32.synth_auc),
                xytext=(h32.ethics_auc + 0.03, h32.synth_auc + 0.06), fontsize=8.5, color=BLUE, fontweight="bold")
    ax.annotate("honesty @47\n(both datasets: inverted)", (h47.ethics_auc, h47.synth_auc),
                xytext=(h47.ethics_auc - 0.22, h47.synth_auc - 0.11), fontsize=8.5, color=BLUE, fontweight="bold")

    ax.set_xlim(0.2, 0.85); ax.set_ylim(0.2, 0.85)
    ax.set_xlabel("AUC on ETHICS (204 items)", fontsize=10.5)
    ax.set_ylabel("AUC on synthetic bank (160 items)", fontsize=10.5)
    ax.set_title("Own-trait vs. other-trait discrimination\nreplicates across two independent datasets",
                 fontsize=11.5, fontweight="bold")

    # legend for significance markers (grey, trait-color-agnostic)
    from matplotlib.lines import Line2D
    sig_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6B7280", markeredgecolor="#1F2937",
               markeredgewidth=1.3, markersize=12, label="Significant in both datasets"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6B7280", markeredgecolor="#9CA3AF",
               alpha=0.85, markersize=10, label="Significant in one dataset"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6B7280", markeredgecolor="#D1D5DB",
               alpha=0.45, markersize=8, label="Not significant"),
    ]
    leg = ax.legend(handles=sig_legend, fontsize=7.8, loc="lower right", framealpha=1.0)
    ax.add_artist(leg)

    trait_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=t.capitalize())
                     for t, c in trait_colors.items()]
    ax.legend(handles=trait_legend, fontsize=8, loc="upper left", framealpha=1.0, title="Trait", title_fontsize=8.5)
    ax.add_artist(leg)

    ax.grid(True, alpha=0.25, zorder=0)
    fig.text(0.01, 0.01, "10,000-permutation test per point; p<0.05 = significant.", fontsize=7.5, color=GREY, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_salience_replication.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_replication()
    fig2_virtue_axis()
    fig3_not_noise()
    fig4_salience_replication()
    print(f"Saved 4 figures (with error bars) to {OUT_DIR}")

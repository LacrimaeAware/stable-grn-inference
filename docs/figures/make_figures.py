"""Generate the README figures from the project's real numbers (no data reload needed).
Run: python docs/figures/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
})
INK, MUTED = "#22303f", "#6b7a8d"
TEAL, CORAL, GOLD, SLATE = "#2a9d8f", "#e76f51", "#e9c46a", "#577399"


# ---- Figure 1: the regime ladder ----
fig, ax = plt.subplots(figsize=(8, 4.4))
labels = ["DREAM4\n(time-series,\nsimulated)", "BEELINE\n(static\nsingle-cell)", "CausalBench / RPE1\n(real CRISPR\ninterventions)"]
centers = [0.92, 0.60, 0.67]
lo = [0.88, 0.50, 0.61]; hi = [0.96, 1.00, 0.70]
colors = [TEAL, GOLD, CORAL]
x = np.arange(3)
ax.bar(x, centers, width=0.6, color=colors, edgecolor=INK, linewidth=0.8, zorder=3)
ax.errorbar(x, centers, yerr=[np.array(centers) - lo, np.array(hi) - centers],
            fmt="none", ecolor=INK, elinewidth=1.4, capsize=6, zorder=4)
ax.axhline(0.5, ls="--", color=MUTED, lw=1.2, zorder=2)
ax.text(2.45, 0.515, "chance (can't orient)", color=MUTED, fontsize=9, ha="right")
for xi, c in zip(x, centers):
    ax.text(xi, c + 0.055, f"{c:.2f}", ha="center", fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.35, 1.05); ax.set_ylabel("recoverability of edge direction")
ax.set_title("Direction is only recoverable when the data contains it")
ax.text(0.5, -0.34, "Identifiability is set by the data regime, not the method — the project's main finding.",
        transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED, style="italic")
fig.tight_layout(); fig.savefig(OUT / "fig1_regime_ladder.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---- Figure 2: every idea worked on clean data, dissolved on real ----
fig, ax = plt.subplots(figsize=(8, 4.4))
ideas = ["Direct-vs-chain\n(3rd-gene test)", "Anti-overfit\n(factor atlas)", "Recover wiring\n(inverse)"]
clean = [0.98, 1.00, 0.98]
real = [0.50, 0.48, 0.50]
x = np.arange(3); w = 0.38
ax.bar(x - w/2, clean, w, label="clean synthetic data", color=TEAL, edgecolor=INK, linewidth=0.8, zorder=3)
ax.bar(x + w/2, real, w, label="real RPE1 data", color="#c9ccd1", edgecolor=INK, linewidth=0.8, zorder=3)
ax.axhline(0.5, ls="--", color=MUTED, lw=1.2, zorder=2)
for xi, c, r in zip(x, clean, real):
    ax.text(xi - w/2, c + 0.02, f"{c:.2f}", ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(xi + w/2, r + 0.02, "faint", ha="center", fontsize=9, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(ideas)
ax.set_ylim(0, 1.12); ax.set_ylabel("how well it worked (0.5 = chance)")
ax.set_title("Every clever idea worked on clean data — and dissolved on the real thing")
ax.legend(frameon=False, loc="upper right")
ax.text(0.5, -0.3, "Not a method problem: real CRISPR data is dominated by a convergent 'cascade' that drowns the specific signal.",
        transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED, style="italic")
fig.tight_layout(); fig.savefig(OUT / "fig2_clean_vs_real.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---- Figure 3: the cascade (one shared program dominates) ----
fig, ax = plt.subplots(figsize=(7, 4.4), subplot_kw=dict(aspect="equal"))
sizes = [53, 47]
wedges, _ = ax.pie(sizes, colors=[CORAL, "#e7e9ec"], startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
ax.text(0, 0.08, "53%", ha="center", fontsize=26, fontweight="bold", color=CORAL)
ax.text(0, -0.16, "one shared\nprogram", ha="center", fontsize=10, color=MUTED)
ax.set_title("Why real data is hard: the cell-cycle 'whirlpool'", pad=14)
ax.text(0, -1.32,
        "Knock out almost ANY essential gene -> the cell runs the same damage program.\n"
        "One convergent 'cell-cycle' mode (CCNB1, MCM3, RRM2, DNMT1, H2AFZ...) is 53% of\n"
        "every response. The specific A->B wiring is a tiny signal buried underneath it.",
        ha="center", fontsize=9.2, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig3_cascade.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("wrote fig1_regime_ladder.png, fig2_clean_vs_real.png, fig3_cascade.png")

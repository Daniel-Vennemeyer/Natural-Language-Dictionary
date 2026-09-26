#!/usr/bin/env python3
"""ICLR-style double-column (figure*) case-study figure: learned taxonomy vs expert
taxonomy on human-annotation recovery (OEQ) and NTA-verdict transfer (AITA), K=3..9.
Labels panels use the NAMED-atom labels; dirs panels use the atom directions."""
import matplotlib as mpl
import matplotlib.pyplot as plt

C_LEARN, C_EXPERT = "#2a78d6", "#eb6834"
C_RAND, C_CEIL, C_CHANCE = "#8a8984", "#222222", "#c9c8c2"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

K = [3, 5, 7, 9]
oeq_labels = {"learned": [.726, .774, .737, .773], "expert": .706,
              "rand": [.535, .567, .473, .634]}
oeq_dirs = {"learned": [.708, .787, .765, .793], "expert": .745,
            "rand": [.641, .669, .690, .706], "rand2sd": [.072, .059, .048, .049],
            "ceiling": .800}
aita_labels = {"learned": [.570, .777, .788, .852], "expert": .765,
               "rand": [.660, .645, .562, .607]}
aita_dirs = {"learned": [.659, .809, .812, .789], "expert": .700,
             "rand": [.617, .669, .702, .702], "rand2sd": [.145, .119, .090, .086]}

fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), sharey=True)
LW, MS = 1.2, 3.6

def _learn(ax, ys):
    ax.plot(K, ys, color=C_LEARN, lw=LW, marker="o", ms=MS,
            markerfacecolor="white", markeredgewidth=0.9, zorder=3)

def _const(ax, v, color, ls="-"):
    ax.plot([K[0], K[-1]], [v, v], color=color, lw=LW, ls=ls, zorder=2)

def _frame(ax, title):
    ax.set_title(title, pad=4)
    ax.set_xticks(K); ax.set_xlim(2.4, 9.6); ax.set_ylim(0.42, 0.88)
    ax.set_xlabel("$K$ (atoms)", labelpad=1.5)
    ax.grid(axis="y", lw=0.4, color="#dddcd6", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0.5, color=C_CHANCE, lw=0.7, zorder=1)

ax = axes[0]
_frame(ax, "(a) OEQ · judged labels")
_const(ax, oeq_dirs["ceiling"], C_CEIL, ls=":")
_learn(ax, oeq_labels["learned"])
_const(ax, oeq_labels["expert"], C_EXPERT)
ax.plot(K, oeq_labels["rand"], color=C_RAND, lw=1.0, ls="--", marker=".", ms=3, zorder=2)
ax.set_ylabel("AUROC")

ax = axes[1]
_frame(ax, "(b) OEQ · directions")
_const(ax, oeq_dirs["ceiling"], C_CEIL, ls=":")
ax.annotate("supervised ceiling", (3.0, oeq_dirs["ceiling"]), xytext=(0, 3),
            textcoords="offset points", fontsize=6.3, color="#52514e")
_learn(ax, oeq_dirs["learned"])
_const(ax, oeq_dirs["expert"], C_EXPERT)
lo = [m - s for m, s in zip(oeq_dirs["rand"], oeq_dirs["rand2sd"])]
hi = [m + s for m, s in zip(oeq_dirs["rand"], oeq_dirs["rand2sd"])]
ax.fill_between(K, lo, hi, color=C_RAND, alpha=0.18, lw=0, zorder=1)
ax.plot(K, oeq_dirs["rand"], color=C_RAND, lw=1.0, ls="--", marker=".", ms=3, zorder=2)

ax = axes[2]
_frame(ax, "(c) AITA · judged labels (transfer)")
_learn(ax, aita_labels["learned"])
_const(ax, aita_labels["expert"], C_EXPERT)
ax.plot(K, aita_labels["rand"], color=C_RAND, lw=1.0, ls="--", marker=".", ms=3, zorder=2)

ax = axes[3]
_frame(ax, "(d) AITA · dirs (judge-indep.)")
_learn(ax, aita_dirs["learned"])
_const(ax, aita_dirs["expert"], C_EXPERT)
lo = [m - s for m, s in zip(aita_dirs["rand"], aita_dirs["rand2sd"])]
hi = [m + s for m, s in zip(aita_dirs["rand"], aita_dirs["rand2sd"])]
ax.fill_between(K, lo, hi, color=C_RAND, alpha=0.18, lw=0, zorder=1)
ax.plot(K, aita_dirs["rand"], color=C_RAND, lw=1.0, ls="--", marker=".", ms=3, zorder=2)

handles = [
    plt.Line2D([], [], color=C_LEARN, lw=LW, marker="o", ms=MS, markerfacecolor="white",
               markeredgewidth=0.9, label="Learned taxonomy"),
    plt.Line2D([], [], color=C_EXPERT, lw=LW, label="Expert taxonomy"),
    plt.Line2D([], [], color=C_RAND, lw=1.0, ls="--", label="Random baseline (±2sd)"),
    plt.Line2D([], [], color=C_CEIL, lw=LW, ls=":", label="Human-supervised ceiling"),
]
fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14),
           frameon=False, columnspacing=1.6, handlelength=1.8, handletextpad=0.5)
fig.subplots_adjust(left=0.065, right=0.995, top=0.82, bottom=0.19, wspace=0.10)
out = "case_study_fig.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)

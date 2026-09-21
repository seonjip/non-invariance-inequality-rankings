"""Figure 1: rank of relative inequality on the odds-ratio and prevalence-ratio scales.

Outputs
-------
output/figures/figure1.png (400 dpi, two panels)
"""
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES, INDICATORS, OUT
from inequality import magnitude

SHORT = {
    "edentulous": "Complete tooth loss",
    "denture_need": "Removable denture need",
    "lt20teeth": "Fewer than 20 natural teeth",
    "chew_disc": "Chewing difficulty",
    "unmet": "Unmet dental care need",
    "untreated": "Untreated caries",
    "perio": "Periodontal pockets (CPI 3-4)",
    "toothpain": "Toothache",
    "poor_sroh": "Poor self-rated oral health",
    "no_checkup": "No dental check-up",
    "caries_exp": "Any caries experience",
}


def ranks(point: dict, measure: str) -> dict:
    values = magnitude([point[i][measure] for i in INDICATORS])
    order = np.argsort(-values)
    return {INDICATORS[i]: position + 1 for position, i in enumerate(order)}


def panel(ax, point: dict, title: str) -> None:
    left, right = ranks(point, "conditional_or"), ranks(point, "prevalence_ratio")
    for indicator in INDICATORS:
        a, b = left[indicator], right[indicator]
        moved = abs(a - b) >= 2
        colour = "#b2182b" if moved else "#9e9e9e"
        ax.plot([0, 1], [a, b], "-o", ms=4, lw=2.2 if moved else 0.9, color=colour,
                zorder=4 if moved else 2)
        ax.text(-0.06, a, f"{SHORT[indicator]} ({point[indicator]['prevalence']:.0f}%)",
                ha="right", va="center", fontsize=7.6, color=colour if moved else "#333333")
        ax.text(1.06, b, SHORT[indicator], ha="left", va="center", fontsize=7.6,
                color=colour if moved else "#333333")
    ax.text(-0.06, -0.45, "Odds-ratio RII", ha="right", va="center", fontsize=9, fontweight="bold")
    ax.text(1.06, -0.45, "Prevalence-ratio RII", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-2.05, 2.15)
    ax.set_ylim(len(INDICATORS) + 0.6, -1.05)
    ax.set_xticks([])
    ax.set_yticks(range(1, len(INDICATORS) + 1))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Rank of relative inequality (1 = largest)", fontsize=8.5, labelpad=6)
    ax.set_title(title, fontsize=9.5, loc="left", pad=14)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def main() -> None:
    with open(OUT / "05_bootstrap_income.pkl", "rb") as handle:
        income = pickle.load(handle)
    with open(OUT / "05_bootstrap_education.pkl", "rb") as handle:
        education = pickle.load(handle)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.1),
                             gridspec_kw={"wspace": 0.62, "left": 0.16, "right": 0.97,
                                          "top": 0.88, "bottom": 0.05})
    panel(axes[0], income["point"], "(a) Household income rank, adults aged 19 years and over")
    panel(axes[1], education["point"], "(b) Educational attainment rank, adults aged 30 years and over")
    fig.savefig(FIGURES / "figure1.png", dpi=400, bbox_inches="tight")
    print("figure written to", FIGURES / "figure1.png")


if __name__ == "__main__":
    main()

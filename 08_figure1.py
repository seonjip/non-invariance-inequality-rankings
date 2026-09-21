"""Figure 1: rank of relative inequality on the three relative estimands.

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
    """Three columns: conditional odds ratio -> marginal odds ratio -> prevalence ratio."""
    x = [0, 1.35, 2.7]
    rank_c = ranks(point, "conditional_or")
    rank_m = ranks(point, "marginal_or")
    rank_p = ranks(point, "prevalence_ratio")
    k = len(INDICATORS)
    for indicator in INDICATORS:
        y = [rank_c[indicator], rank_m[indicator], rank_p[indicator]]
        moved = max(abs(y[0] - y[1]), abs(y[1] - y[2]), abs(y[0] - y[2])) >= 2
        colour = "#b2182b" if moved else "#9e9e9e"
        ax.plot(x, y, "-o", ms=4, lw=2.0 if moved else 0.9, color=colour, zorder=4 if moved else 2)
        ax.text(x[0] - 0.1, y[0], f"{SHORT[indicator]} ({point[indicator]['prevalence']:.0f}%)",
                ha="right", va="center", fontsize=7.4, color=colour if moved else "#333333")
        ax.text(x[2] + 0.1, y[2], SHORT[indicator], ha="left", va="center", fontsize=7.4,
                color=colour if moved else "#333333")
    for rank in range(1, k + 1):                      # faint rank numbers for reading exact ranks
        ax.text(-3.2, rank, str(rank), ha="center", va="center", fontsize=7.2, color="#8a8a8a")
    ax.text(-3.2, -0.6, "Rank", ha="center", va="center", fontsize=7.6, color="#6a6a6a",
            fontweight="bold")
    for xi, label in zip(x, ["Conditional\nodds ratio", "Marginal\nodds ratio", "Prevalence\nratio"]):
        ax.text(xi, -0.6, label, ha="center", va="center", fontsize=8.4, fontweight="bold")
        ax.axvline(xi, color="#eeeeee", lw=0.6, zorder=0)
    ax.text((x[0] + x[1]) / 2, k + 0.75, "non-\ncollapsibility", ha="center", va="center",
            fontsize=7.2, style="italic", color="#555555")
    ax.text((x[1] + x[2]) / 2, k + 0.75, "change\nof scale", ha="center", va="center",
            fontsize=7.2, style="italic", color="#555555")
    ax.set_xlim(-3.45, 5.0)
    ax.set_ylim(k + 1.1, -1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, loc="left", pad=16)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def main() -> None:
    with open(OUT / "05_bootstrap_income.pkl", "rb") as handle:
        income = pickle.load(handle)
    with open(OUT / "05_bootstrap_education.pkl", "rb") as handle:
        education = pickle.load(handle)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(13.56, 6.4),
                             gridspec_kw={"wspace": 0.217, "left": 0.12, "right": 0.985,
                                          "top": 0.86, "bottom": 0.04})
    panel(axes[0], income["point"], "(a) Household income rank, adults aged 19 years and over")
    panel(axes[1], education["point"], "(b) Educational attainment rank, adults aged 30 years and over")
    fig.savefig(FIGURES / "figure1.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "figure1.tif", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    print("figure written to", FIGURES / "figure1.png")


if __name__ == "__main__":
    main()

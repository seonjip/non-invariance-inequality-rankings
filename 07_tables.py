"""Write Tables 1-4 and Supplementary Tables S1-S10 as CSV files.

Every table is generated from the stored replicate sets, so confidence intervals for the
same indicator are identical wherever that indicator appears.

Outputs
-------
output/tables/*.csv
"""
import pickle

import numpy as np
import pandas as pd

from config import CLINICAL, INDICATORS, LABELS, OUT, TABLES, THRESHOLD_SET
from inequality import MEASURES, magnitude


def load(name: str) -> dict:
    with open(OUT / f"05_bootstrap_{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def ci(draws: np.ndarray, measure: str, digits: int = 2) -> str:
    column = draws[:, MEASURES.index(measure)]
    low, high = np.percentile(column, [2.5, 97.5])
    return f"({low:.{digits}f}, {high:.{digits}f})"


def estimate(point: dict, draws: np.ndarray, measure: str, digits: int = 2) -> str:
    return f"{point[measure]:.{digits}f} {ci(draws, measure, digits)}"


def main_table(result: dict, indicators) -> pd.DataFrame:
    rows = []
    for indicator in sorted(indicators, key=lambda i: result["point"][i]["prevalence"]):
        p, d = result["point"][indicator], result["draws"][indicator]
        rows.append({
            "Indicator": LABELS.get(indicator, indicator),
            "Prevalence, % (n)": f"{p['prevalence']:.1f} ({p['n']})",
            "Standardised prevalence, advantaged / disadvantaged":
                f"{p['p_advantaged']:.1f} / {p['p_disadvantaged']:.1f}",
            "Conditional OR (95% CI)": estimate(p, d, "conditional_or"),
            "Marginal OR (95% CI)": estimate(p, d, "marginal_or"),
            "Prevalence ratio (95% CI)": estimate(p, d, "prevalence_ratio"),
            "SII (95% CI)": estimate(p, d, "slope_index", 1),
        })
    return pd.DataFrame(rows)


def agreement_table(income: dict, education: dict, favourable: dict) -> pd.DataFrame:
    def row(axis, label, stats):
        return {"Socioeconomic axis and indicator set": axis, "Comparison of measures": label,
                "Spearman correlation (95% CI)":
                    f"{stats['rho']:.2f} ({stats['rho_ci'][0]:.2f}, {stats['rho_ci'][1]:.2f})",
                "Discordant pairs": f"{stats['discordant_pairs']} of {stats['n_pairs']}",
                "Replicates with imperfect concordance, %": f"{stats['pct_replicates_imperfect']:.1f}"}

    rows = []
    for axis, result in (("Household income", income), ("Educational attainment", education)):
        a = result["agreement"]
        rows += [
            row(axis, "Conditional vs marginal odds ratio (non-collapsibility)", a["conditional_vs_marginal_or"]),
            row(axis, "Marginal odds ratio vs prevalence ratio (scale)", a["marginal_or_vs_prevalence_ratio"]),
            row(axis, "Conditional odds ratio vs prevalence ratio (combined)", a["conditional_or_vs_prevalence_ratio"]),
            row(axis, "Conditional odds ratio vs slope index", a["conditional_or_vs_slope_index"]),
            row(f"{axis}, 6 clinical indicators", "Conditional odds ratio vs prevalence ratio", a["clinical_only"]),
        ]
    rows.append(row("Household income, 10 threshold variants",
                    "Conditional odds ratio vs prevalence ratio",
                    income["agreement"]["threshold_set"]))

    adverse = magnitude([income["point"][i]["prevalence_ratio"] for i in INDICATORS])
    favour = magnitude([favourable["point"]["fav_" + i]["prevalence_ratio"] for i in INDICATORS])
    import itertools
    from scipy.stats import spearmanr
    pairs = list(itertools.combinations(range(len(INDICATORS)), 2))
    discordant = sum(1 for i, j in pairs
                     if np.sign(adverse[i] - adverse[j]) != np.sign(favour[i] - favour[j]))
    rows.append({"Socioeconomic axis and indicator set": "Household income, favourable coding",
                 "Comparison of measures": "Prevalence ratio, adverse vs favourable coding",
                 "Spearman correlation (95% CI)": f"{spearmanr(adverse, favour).statistic:.2f}",
                 "Discordant pairs": f"{discordant} of {len(pairs)}",
                 "Replicates with imperfect concordance, %": "-"})
    return pd.DataFrame(rows)


def main() -> None:
    income, education, favourable = load("income"), load("education"), load("favourable")

    main_table(income, INDICATORS).to_csv(TABLES / "table1_income.csv", index=False)
    main_table(education, INDICATORS).to_csv(TABLES / "table2_education.csv", index=False)
    agreement_table(income, education, favourable).to_csv(TABLES / "table3_agreement.csv", index=False)
    main_table(income, THRESHOLD_SET).to_csv(TABLES / "table4_thresholds.csv", index=False)

    # Supplementary tables drawing on the same replicate sets and on 06_sensitivity.pkl
    with open(OUT / "06_sensitivity.pkl", "rb") as handle:
        sensitivity = pickle.load(handle)

    pd.DataFrame([{"Indicator": LABELS[i], "Definition in the survey data": ""}
                  for i in INDICATORS]).to_csv(TABLES / "tableS1_definitions.csv", index=False)

    rows = []
    for indicator in sorted(INDICATORS, key=lambda i: income["point"][i]["prevalence"]):
        p = income["point"][indicator]
        rows.append({"Indicator": LABELS[indicator], "Prevalence, %": f"{p['prevalence']:.1f}",
                     "Conditional OR": f"{p['conditional_or']:.2f}",
                     "Marginal OR": f"{p['marginal_or']:.2f}",
                     "Prevalence ratio": f"{p['prevalence_ratio']:.2f}",
                     "Non-collapsibility component": f"{p['noncollapsibility']:.2f}",
                     "Scale component": f"{p['scale_component']:.2f}"})
    pd.DataFrame(rows).to_csv(TABLES / "tableS4_decomposition_income.csv", index=False)

    rows = []
    for indicator in sorted(INDICATORS, key=lambda i: income["point"][i]["prevalence"]):
        adverse = income["point"][indicator]
        favour = favourable["point"]["fav_" + indicator]
        rows.append({"Indicator (adverse coding)": LABELS[indicator],
                     "Prevalence, %": f"{adverse['prevalence']:.1f}",
                     "Prevalence ratio, adverse coding": f"{adverse['prevalence_ratio']:.2f}",
                     "Prevalence ratio, favourable coding": f"{favour['prevalence_ratio']:.2f}"})
    pd.DataFrame(rows).to_csv(TABLES / "tableS5_favourable_coding.csv", index=False)

    rows = []
    for label, result in sensitivity["age_groups"].items():
        for indicator, values in result["point"].items():
            rows.append({"Age group, years": label, "Indicator": LABELS[indicator],
                         "Prevalence, %": f"{values['prevalence']:.1f}",
                         "Conditional OR": f"{values['conditional_or']:.2f}",
                         "Prevalence ratio": f"{values['prevalence_ratio']:.2f}",
                         "Ratio of the two":
                             f"{values['conditional_or'] / values['prevalence_ratio']:.2f}"})
    pd.DataFrame(rows).to_csv(TABLES / "tableS6_age_groups.csv", index=False)

    print("tables written to", TABLES)


if __name__ == "__main__":
    main()

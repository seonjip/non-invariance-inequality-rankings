"""Write Tables 1-4 and Supplementary Tables S1-S11 as CSV files.

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
from inequality import MEASURES, favourable_prevalence_ratio, magnitude


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
            "Weighted prevalence, % (analytic n)": f"{p['prevalence']:.1f} ({p['n']})",
            "Standardised prevalence at rank 0 / rank 1, %":
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

    # favourable coding computed within the same replicates from the standardised prevalences
    adverse = magnitude([income["point"][i]["prevalence_ratio"] for i in INDICATORS])
    favour = magnitude([favourable_prevalence_ratio(income["point"][i]["p_advantaged"],
                                                    income["point"][i]["p_disadvantaged"]) for i in INDICATORS])
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


def agreement_row(label: str, stats: dict) -> dict:
    return {"Specification": label,
            "Spearman correlation (95% CI)":
                f"{stats['rho']:.2f} ({stats['rho_ci'][0]:.2f}, {stats['rho_ci'][1]:.2f})",
            "Discordant pairs": f"{stats['discordant_pairs']} of {stats['n_pairs']}",
            "Replicates with imperfect concordance, %": f"{stats['pct_replicates_imperfect']:.1f}"}


def reversal_grid() -> pd.DataFrame:
    """Supplementary Table S3: prevalence ratio a more prevalent comparator needs to attain the
    same MARGINAL odds ratio as a less prevalent reference indicator (analytic; no data)."""
    references = [(0.05, 2.00), (0.20, 2.00), (0.05, 1.50), (0.20, 1.50), (0.40, 1.50)]
    comparators = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    rows = []
    for p0, pr in references:
        p1 = p0 * pr
        marginal_or = (p1 / (1 - p1)) / (p0 / (1 - p0))
        row = {"Reference indicator": f"p0={p0:.0%}, PR={pr:.2f} (marginal OR={marginal_or:.2f})"}
        for q0 in comparators:
            if q0 <= p0:
                row[f"{q0:.0%}"] = "-"
                continue
            odds = q0 / (1 - q0) * marginal_or
            q1 = odds / (1 + odds)
            row[f"{q0:.0%}"] = f"{q1 / q0:.2f}" if q1 / q0 <= 1 / q0 else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def supplementary_tables(income: dict, sensitivity: dict) -> None:
    # S2: common complete-case sample
    cc = sensitivity["complete_case"]
    n = max(v["n"] for v in cc["point"].values())
    row = agreement_row("Ten indicators not requiring the periodontal examination", cc["agreement"])
    row["n"] = n
    pd.DataFrame([row]).to_csv(TABLES / "tableS2_complete_case.csv", index=False)

    # S3: analytic reversal grid
    reversal_grid().to_csv(TABLES / "tableS3_reversal_grid.csv", index=False)

    # S7: covariate sets
    names = {"M0": "Model 0, unadjusted", "M1": "Model 1, age, age squared and sex",
             "M2": "Model 2, Model 1 plus urbanicity and housing type"}
    pd.DataFrame([agreement_row(names[k], v["agreement"])
                  for k, v in sensitivity["covariate_sets"].items()]).to_csv(
        TABLES / "tableS7_covariate_sets.csv", index=False)

    # S8: alternative socioeconomic rank definitions
    names = {"income_q4": "Household income quartiles", "income_d10": "Household income deciles",
             "income_individual_q5": "Individual income quintiles",
             "education_4": "Educational attainment, four categories",
             "education_3": "Educational attainment, three categories"}
    rows = [agreement_row(names[k], v["agreement"]) for k, v in sensitivity["sep_variants"].items()]
    rows.append(agreement_row("Household income quintiles, ages 30 years and over",
                              sensitivity["income_age30plus"]["agreement"]))
    pd.DataFrame(rows).to_csv(TABLES / "tableS8_rank_definitions.csv", index=False)

    # S9: monotonicity and quadratic rank
    quad = sensitivity["quadratic_rank"]["point"]
    rows = []
    for indicator in sorted(INDICATORS, key=lambda i: income["point"][i]["prevalence"]):
        lin = income["point"][indicator]
        rows.append({"Indicator": LABELS[indicator],
                     "Prevalence changes monotonically across income quintiles":
                         "Yes" if sensitivity["monotonic"][indicator] else "No",
                     "Conditional OR, linear rank": f"{lin['conditional_or']:.2f}",
                     "Conditional OR, quadratic rank": f"{quad[indicator]['conditional_or']:.2f}",
                     "Prevalence ratio, linear rank": f"{lin['prevalence_ratio']:.2f}",
                     "Prevalence ratio, quadratic rank": f"{quad[indicator]['prevalence_ratio']:.2f}"})
    pd.DataFrame(rows).to_csv(TABLES / "tableS9_quadratic_monotonicity.csv", index=False)

    # S10: indices based on predicted probabilities
    rows = []
    for indicator in sorted(INDICATORS, key=lambda i: income["point"][i]["prevalence"]):
        p, d = income["point"][indicator], income["draws"][indicator]
        rows.append({"Indicator": LABELS[indicator], "Prevalence, %": f"{p['prevalence']:.1f}",
                     "Conditional OR": f"{p['conditional_or']:.2f}",
                     "Prevalence ratio": f"{p['prevalence_ratio']:.2f}",
                     "Predicted-probability ratio at covariate means (95% CI)":
                         estimate(p, d, "pred_at_means"),
                     "Unadjusted predicted-probability ratio": f"{p['unadjusted_pred_ratio']:.2f}"})
    pd.DataFrame(rows).to_csv(TABLES / "tableS10_predicted_probability.csv", index=False)


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
                         "Marginal OR": f"{values['marginal_or']:.2f}",
                         "Prevalence ratio": f"{values['prevalence_ratio']:.2f}",
                         "Non-collapsibility component": f"{values['noncollapsibility']:.2f}",
                         "Scale component": f"{values['scale_component']:.2f}"})
    pd.DataFrame(rows).to_csv(TABLES / "tableS6_age_groups.csv", index=False)

    rows = []
    ia, ip = MEASURES.index("conditional_or"), None
    for indicator in INDICATORS:
        estimate_, low, high, se = sensitivity["taylor"][indicator]
        boot = income["draws"][indicator][:, ia]
        b_low, b_high = np.percentile(boot, [2.5, 97.5])
        rows.append({"Indicator": LABELS[indicator], "Conditional OR": f"{estimate_:.2f}",
                     "Taylor-linearised 95% CI": f"({low:.2f}, {high:.2f})",
                     "Rao-Wu bootstrap 95% CI": f"({b_low:.2f}, {b_high:.2f})",
                     "Ratio of standard errors": f"{np.std(np.log(boot), ddof=1) / se:.2f}"})
    pd.DataFrame(rows).to_csv(TABLES / "tableS11_taylor_vs_bootstrap.csv", index=False)

    supplementary_tables(income, sensitivity)
    print("tables written to", TABLES)


if __name__ == "__main__":
    main()

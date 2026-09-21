"""Design-based bootstrap: 2000 replicates of the whole estimation sequence.

Primary sampling units are resampled with replacement within variance estimation strata
using the Rao-Wu rescaling bootstrap: in a stratum with n observed units, n - 1 units are
drawn and the survey weights of the selected units are multiplied by n / (n - 1).
The rank transformation, the model fit, the standardisation, the computation of every
estimand and the ranking of indicators are all repeated inside each replicate, so the
agreement statistics reflect sampling variability in the ranking itself as well as the
dependence between indicators measured on the same participants.

Outputs
-------
output/05_bootstrap_<axis>.pkl : point estimates, replicate draws and agreement statistics
"""
import pickle

import numpy as np
import pandas as pd

from config import CLINICAL, INDICATORS, N_BOOT, OUT, SEEDS, SEP, THRESHOLD_SET
from inequality import MEASURES, agreement, estimands, fractional_rank, pairwise_reversal


def _covariates(sample: pd.DataFrame, names) -> list[np.ndarray]:
    lookup = {
        "age": sample.age.values.astype(float),
        "age2": sample.age2.values.astype(float),
        "sex": (sample.sex.values == 2).astype(float),
        "town": (sample.town_t.values == 2).astype(float),
        "apt": (sample.apt_t.values == 2).astype(float),
    }
    return [lookup[n] for n in names]


def run(d: pd.DataFrame, indicators, sep_name: str, seed: int,
        covset=("age", "age2", "sex"), n_boot: int = N_BOOT,
        quadratic_rank: bool = False) -> dict:
    variable, levels, min_age = SEP[sep_name]
    frame = d[(d.age >= min_age) & d[variable].notna()].reset_index(drop=True)

    weight = frame.wt_oe.values.astype(float)
    code = frame[variable].values.astype(float)
    outcome = {i: frame[i].values.astype(float) for i in indicators}
    covariates = {name: column for name, column in zip(covset, _covariates(frame, covset))}

    def one(index: np.ndarray, multiplier: np.ndarray):
        out = {}
        for indicator in indicators:
            recorded = ~np.isnan(outcome[indicator][index])
            keep = index[recorded]
            w = weight[keep] * multiplier[recorded]
            rank = fractional_rank(code[keep], w, levels)
            result = estimands(outcome[indicator][keep], w, rank,
                               [covariates[n][keep] for n in covset],
                               quadratic_rank=quadratic_rank)
            if result is None:
                return None
            out[indicator] = result
        return out

    point = one(np.arange(len(frame)), np.ones(len(frame)))
    if point is None:
        raise RuntimeError("model did not converge in the full sample")

    # primary sampling units grouped by variance estimation stratum
    psu_index = frame.groupby(["kstrata", "psu"]).indices
    strata: dict = {}
    for (stratum, _), rows in psu_index.items():
        strata.setdefault(stratum, []).append(rows)

    rng = np.random.default_rng(seed)
    draws = {i: [] for i in indicators}
    failures = 0
    for _ in range(n_boot):
        parts, multipliers = [], []
        for units in strata.values():
            n_h = len(units)
            for k in rng.integers(0, n_h, n_h - 1):          # Rao-Wu: draw n_h - 1 units
                parts.append(units[k])
                multipliers.append(np.full(len(units[k]), n_h / (n_h - 1)))
        replicate = one(np.concatenate(parts), np.concatenate(multipliers))
        if replicate is None:
            failures += 1
            continue
        for indicator in indicators:
            draws[indicator].append([replicate[indicator][m] for m in MEASURES])

    draws = {i: np.array(v) for i, v in draws.items()}
    return {"point": point, "draws": draws, "indicators": list(indicators),
            "n_boot": n_boot, "failures": failures}


def main() -> None:
    d = pd.read_pickle(OUT / "02_indicators.pkl")

    # one set of replicates covers Tables 1, 3 and 4 on the income axis
    income_set = INDICATORS + [i for i in THRESHOLD_SET if i not in INDICATORS]
    income = run(d, income_set, "income_q5", SEEDS["income_main"])
    education = run(d, INDICATORS, "education_4", SEEDS["education_main"])
    favourable = run(d, ["fav_" + i for i in INDICATORS], "income_q5", SEEDS["favourable_coding"])

    for name, result in (("income", income), ("education", education)):
        summary = {
            "conditional_vs_marginal_or": agreement(result["point"], result["draws"], INDICATORS,
                                                    "conditional_or", "marginal_or"),
            "marginal_or_vs_prevalence_ratio": agreement(result["point"], result["draws"], INDICATORS,
                                                         "marginal_or", "prevalence_ratio"),
            "conditional_or_vs_prevalence_ratio": agreement(result["point"], result["draws"], INDICATORS,
                                                            "conditional_or", "prevalence_ratio"),
            "conditional_or_vs_slope_index": agreement(result["point"], result["draws"], INDICATORS,
                                                       "conditional_or", "slope_index"),
            "clinical_only": agreement(result["point"], result["draws"], CLINICAL,
                                       "conditional_or", "prevalence_ratio"),
            "pred_at_means_vs_prevalence_ratio": agreement(result["point"], result["draws"], INDICATORS,
                                                           "pred_at_means", "prevalence_ratio"),
        }
        result["agreement"] = summary
        result["pairwise_reversal"] = pairwise_reversal(result["draws"], INDICATORS)
        print(f"\n== {name} ==")
        for label, stats in summary.items():
            print(f"  {label:<38} rho {stats['rho']:.2f} "
                  f"({stats['rho_ci'][0]:.2f}, {stats['rho_ci'][1]:.2f})  "
                  f"discordant {stats['discordant_pairs']}/{stats['n_pairs']}")

    income["agreement"]["threshold_set"] = agreement(
        income["point"], income["draws"], THRESHOLD_SET, "conditional_or", "prevalence_ratio")

    for name, result in (("income", income), ("education", education), ("favourable", favourable)):
        with open(OUT / f"05_bootstrap_{name}.pkl", "wb") as handle:
            pickle.dump(result, handle)
        print(f"{name}: {result['n_boot']} replicates, {result['failures']} non-convergent")


if __name__ == "__main__":
    main()

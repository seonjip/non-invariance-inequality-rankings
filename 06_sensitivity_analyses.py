"""Seven sets of sensitivity analyses reported in the supplementary material.

1. alternative indicator thresholds (five pairs, run with the main income replicates)
2. estimation within five age groups
3. three covariate sets
4. five alternative definitions of the socioeconomic rank, and income restricted to ages 30+
5. a quadratic term for the socioeconomic rank, with a monotonicity check
6. indices based on predicted probabilities (at covariate means, and unadjusted)
7. restricted indicator sets: clinical conditions only, and a common complete-case sample

Outputs
-------
output/06_sensitivity.pkl
"""
import pickle

import numpy as np
import pandas as pd

from config import AGE_GROUPS, COVSETS, INDICATORS, N_BOOT, OUT, SEEDS, SEP
from importlib import import_module
from inequality import agreement, fractional_rank, taylor_ci

run = import_module("05_bootstrap").run
MIN_CASES = 30  # an indicator is not estimable in a stratum below this many cases or non-cases


def main() -> None:
    d = pd.read_pickle(OUT / "02_indicators.pkl")
    results: dict = {}

    # 2. age groups -----------------------------------------------------------------
    results["age_groups"] = {}
    for label, low, high in AGE_GROUPS:
        subset = d[(d.age >= low) & (d.age <= high)]
        estimable = [i for i in INDICATORS
                     if np.nansum(subset[i]) >= MIN_CASES
                     and (subset[i].notna().sum() - np.nansum(subset[i])) >= MIN_CASES]
        result = run(subset, estimable, "income_q5", SEEDS["age_groups"] + low)
        result["agreement"] = agreement(result["point"], result["draws"], estimable,
                                        "conditional_or", "prevalence_ratio")
        results["age_groups"][label] = result
        print(f"age {label:<6} n indicators {len(estimable):>2}  "
              f"rho {result['agreement']['rho']:.2f}  "
              f"discordant {result['agreement']['discordant_pairs']}/{result['agreement']['n_pairs']}")

    # 3. covariate sets -------------------------------------------------------------
    results["covariate_sets"] = {}
    for name, covset in COVSETS.items():
        result = run(d, INDICATORS, "income_q5", SEEDS["covsets"], covset=covset)
        result["agreement"] = agreement(result["point"], result["draws"], INDICATORS,
                                        "conditional_or", "prevalence_ratio")
        results["covariate_sets"][name] = result
        print(f"covariates {name}  rho {result['agreement']['rho']:.2f}  "
              f"discordant {result['agreement']['discordant_pairs']}")

    # 4. alternative socioeconomic rank definitions ---------------------------------
    results["sep_variants"] = {}
    for name in ("income_q4", "income_d10", "income_individual_q5", "education_4", "education_3"):
        result = run(d, INDICATORS, name, SEEDS["sep_variants"])
        result["agreement"] = agreement(result["point"], result["draws"], INDICATORS,
                                        "conditional_or", "prevalence_ratio")
        results["sep_variants"][name] = result
        print(f"rank {name:<22} rho {result['agreement']['rho']:.2f}  "
              f"discordant {result['agreement']['discordant_pairs']}")

    # income restricted to the age range used for education
    older = d[d.age >= 30]
    result = run(older, INDICATORS, "income_q5", SEEDS["sep_variants"] + 1)
    result["agreement"] = agreement(result["point"], result["draws"], INDICATORS,
                                    "conditional_or", "prevalence_ratio")
    results["income_age30plus"] = result

    # 5. quadratic rank and monotonicity --------------------------------------------
    quadratic = run(d, INDICATORS, "income_q5", SEEDS["quadratic"], quadratic_rank=True)
    quadratic["agreement"] = agreement(quadratic["point"], quadratic["draws"], INDICATORS,
                                       "conditional_or", "prevalence_ratio")
    results["quadratic_rank"] = quadratic

    monotonic = {}
    eligible = d[(d.age >= 19) & d.ho_incm5.notna()]
    for indicator in INDICATORS:
        sample = eligible[eligible[indicator].notna()]
        prevalence = [np.average(sample.loc[sample.ho_incm5 == q, indicator],
                                 weights=sample.loc[sample.ho_incm5 == q, "wt_oe"])
                      for q in (5, 4, 3, 2, 1)]
        difference = np.diff(prevalence)
        monotonic[indicator] = bool(np.all(difference >= 0) or np.all(difference <= 0))
    results["monotonic"] = monotonic
    print(f"monotonic across income quintiles: {sum(monotonic.values())} of {len(monotonic)}")

    # 7. common complete-case sample (indicators not requiring the periodontal examination)
    no_perio = [i for i in INDICATORS if i != "perio"]
    complete = d.dropna(subset=no_perio + ["ho_incm5", "age", "sex"])
    result = run(complete, no_perio, "income_q5", SEEDS["complete_case"])
    result["agreement"] = agreement(result["point"], result["draws"], no_perio,
                                    "conditional_or", "prevalence_ratio")
    results["complete_case"] = result
    print(f"complete case n = {len(complete)}  rho {result['agreement']['rho']:.2f}  "
          f"discordant {result['agreement']['discordant_pairs']}/{result['agreement']['n_pairs']}")

    # variance check: Taylor-linearised CI for the conditional odds ratio (income axis)
    taylor = {}
    frame = d[(d.age >= 19) & d.ho_incm5.notna()]
    for indicator in INDICATORS:
        s_ = frame[frame[indicator].notna()]
        w = s_.wt_oe.values.astype(float)
        rank = fractional_rank(s_.ho_incm5.values.astype(float), w, [5, 4, 3, 2, 1])
        cov = [s_.age.values.astype(float), s_.age2.values.astype(float), (s_.sex.values == 2).astype(float)]
        taylor[indicator] = taylor_ci(s_[indicator].values.astype(float), w, rank, cov,
                                      s_.kstrata.values, s_.psu.values)
    results["taylor"] = taylor

    with open(OUT / "06_sensitivity.pkl", "wb") as handle:
        pickle.dump(results, handle)


if __name__ == "__main__":
    main()

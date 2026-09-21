"""Point estimates of the three estimands and the slope index for every indicator.

Outputs
-------
output/04_point_estimates.pkl : nested dictionary, axis -> indicator -> estimands
"""
import pickle

import numpy as np
import pandas as pd

from config import COVSETS, INDICATORS, LABELS, OUT, SEP, THRESHOLD_SET
from inequality import estimands, fractional_rank


def covariate_columns(d: pd.DataFrame, names) -> list[np.ndarray]:
    lookup = {
        "age": d.age.values.astype(float),
        "age2": d.age2.values.astype(float),
        "sex": (d.sex.values == 2).astype(float),
        "town": (d.town_t.values == 2).astype(float),
        "apt": (d.apt_t.values == 2).astype(float),
    }
    return [lookup[n] for n in names]


def analyse(d: pd.DataFrame, indicators, sep_name: str, covset=("age", "age2", "sex"),
            quadratic_rank: bool = False) -> dict:
    variable, levels, min_age = SEP[sep_name]
    eligible = d[(d.age >= min_age) & d[variable].notna()]
    out = {}
    for indicator in indicators:
        sample = eligible[eligible[indicator].notna()]
        weight = sample.wt_oe.values.astype(float)
        rank = fractional_rank(sample[variable].values.astype(float), weight, levels)
        result = estimands(sample[indicator].values.astype(float), weight, rank,
                           covariate_columns(sample, covset), quadratic_rank=quadratic_rank)
        if result is None:
            raise RuntimeError(f"model did not converge for {indicator}")
        out[indicator] = result
    return out


def main() -> None:
    d = pd.read_pickle(OUT / "02_indicators.pkl")
    point = {
        "income": analyse(d, INDICATORS + THRESHOLD_SET[1::2], "income_q5"),
        "education": analyse(d, INDICATORS, "education_4"),
        "income_favourable": analyse(d, ["fav_" + i for i in INDICATORS], "income_q5"),
    }

    rows = []
    for indicator, values in point["income"].items():
        rows.append({
            "indicator": LABELS.get(indicator.replace("fav_", ""), indicator),
            "prevalence": values["prevalence"],
            "conditional_or": values["conditional_or"],
            "marginal_or": values["marginal_or"],
            "prevalence_ratio": values["prevalence_ratio"],
            "slope_index": values["slope_index"],
            "noncollapsibility": values["noncollapsibility"],
            "scale_component": values["scale_component"],
        })
    table = pd.DataFrame(rows).sort_values("prevalence")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    with open(OUT / "04_point_estimates.pkl", "wb") as handle:
        pickle.dump(point, handle)


if __name__ == "__main__":
    main()

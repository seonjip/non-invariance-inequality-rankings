"""Construct the 11 binary indicators, the five threshold variants and the favourable recoding.

Outputs
-------
output/02_indicators.pkl
"""
import numpy as np
import pandas as pd

from config import INDICATORS, OUT, THRESHOLD_VARIANTS

CPI_SEXTANTS = ["O_CPI_UR", "O_CPI_UM", "O_CPI_UL", "O_CPI_LR", "O_CPI_LM", "O_CPI_LL"]


def binary(series: pd.Series, one, zero) -> pd.Series:
    """Recode selected response codes to 1 and 0, leaving every other code missing."""
    out = pd.Series(np.nan, index=series.index)
    out[series.isin(one)] = 1.0
    out[series.isin(zero)] = 0.0
    return out


def main() -> None:
    d = pd.read_pickle(OUT / "01_sample.pkl")

    # --- the eleven indicators -------------------------------------------------------
    d["edentulous"] = np.where(d.exam, (d.n_teeth == 0).astype(float), np.nan)
    d["lt20teeth"] = np.where(d.exam, (d.n_teeth < 20).astype(float), np.nan)
    d["denture_need"] = np.where(
        d.exam, ((d.O_DENT_U.isin([1, 2])) | (d.O_DENT_L.isin([1, 2]))).astype(float), np.nan)
    d["untreated"] = d.O_DIP                      # one or more decayed permanent teeth
    d["caries_exp"] = d.O_DMFIP                   # DMFT of one or more
    d["perio"] = d.NO_CPI_34                      # community periodontal index score 3 or 4
    d["poor_sroh"] = binary(d.OR1, [4, 5], [1, 2, 3])
    d["chew_disc"] = d.O_chew_d
    d["toothpain"] = binary(d.O_pain, [1], [0])
    d["unmet"] = binary(d.BM14, [1], [2, 3])
    d["no_checkup"] = binary(d.OR1_2, [0], [1])

    # --- five alternative thresholds of the same constructs -------------------------
    d["lt25teeth"] = np.where(d.exam, (d.n_teeth < 25).astype(float), np.nan)
    d["sroh_broad"] = binary(d.OR1, [3, 4, 5], [1, 2])
    cpi = d[CPI_SEXTANTS]
    recorded = cpi.isin([0, 1, 2, 3, 4]).any(axis=1)
    d["cpi4"] = np.where(recorded, cpi.isin([4]).any(axis=1).astype(float), np.nan)
    d["untreated2"] = np.where(d.O_DTP.notna(), (d.O_DTP >= 2).astype(float), np.nan)
    d["chew_broad"] = binary(d.BM7, [1, 2, 3], [4, 5])

    # --- favourable (complementary) coding of every indicator -----------------------
    for indicator in INDICATORS:
        d["fav_" + indicator] = np.where(d[indicator].notna(), 1 - d[indicator], np.nan)

    # --- alternative socioeconomic variables ----------------------------------------
    d["edu3"] = d.edu.map({1: 1, 2: 2, 3: 2, 4: 3})
    income = d.ainc.copy()
    weight = d.wt_oe.where(income.notna())
    order = np.argsort(income.where(income.notna()).values)
    cumulative = np.nancumsum(weight.values[order]) / np.nansum(weight.values)
    decile = np.full(len(d), np.nan)
    for position, row in enumerate(order):
        if np.isnan(income.values[row]):
            continue
        decile[row] = min(10, int(cumulative[position] * 10) + 1)
    d["inc_dec"] = decile

    for indicator in INDICATORS + THRESHOLD_VARIANTS:
        recorded = d[indicator].notna()
        prevalence = 100 * np.average(d.loc[recorded, indicator], weights=d.loc[recorded, "wt_oe"])
        print(f"{indicator:<14} n = {int(recorded.sum()):>6}   weighted prevalence = {prevalence:5.1f}%")

    d.to_pickle(OUT / "02_indicators.pkl")


if __name__ == "__main__":
    main()

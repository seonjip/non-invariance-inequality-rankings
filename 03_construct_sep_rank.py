"""Build the weighted fractional (ridit) socioeconomic rank for every definition.

The rank is recomputed inside each bootstrap replicate in 05_bootstrap.py; this script
writes the full-sample ranks so that they can be inspected and so that the distribution of
each definition can be checked before estimation.

Outputs
-------
output/03_sep_ranks.pkl
"""
import numpy as np
import pandas as pd

from config import OUT, SEP
from inequality import fractional_rank


def main() -> None:
    d = pd.read_pickle(OUT / "02_indicators.pkl")
    ranks = pd.DataFrame(index=d.index)

    for name, (variable, levels, min_age) in SEP.items():
        eligible = (d.age >= min_age) & d[variable].notna()
        weight = d.loc[eligible, "wt_oe"].values.astype(float)
        code = d.loc[eligible, variable].values.astype(float)
        column = np.full(len(d), np.nan)
        column[np.where(eligible)[0]] = fractional_rank(code, weight, levels)
        ranks[name] = column
        share = pd.Series(code).value_counts(normalize=True).sort_index()
        print(f"{name:<22} n = {int(eligible.sum()):>6}   "
              f"rank range {np.nanmin(column):.3f} to {np.nanmax(column):.3f}   "
              f"categories = {len(share)}")

    ranks.to_pickle(OUT / "03_sep_ranks.pkl")


if __name__ == "__main__":
    main()

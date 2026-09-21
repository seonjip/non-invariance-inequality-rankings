"""Read the KNHANES VII oral examination file and define the analytic samples.

Outputs
-------
output/01_sample.pkl : one row per participant aged 19 years and over who received the
    oral examination, with the derived tooth counts and the survey design variables.
"""
import numpy as np
import pandas as pd
import pyreadstat

from config import RAW, OUT

PERMANENT_TEETH = [f"{quadrant}{position}" for quadrant in (1, 2, 3, 4) for position in range(1, 9)]
PRESENT_CODES = [0, 1, 3, 6, 7]   # sound, decayed, filled, sealed, non-carious restored
MISSING_CODES = [4, 5]            # missing because of caries, missing for other reasons
UNKNOWN_CODES = [9]               # not recordable


def main() -> None:
    df, _ = pyreadstat.read_sav(RAW)

    present, missing, unknown = [], [], []
    for tooth in PERMANENT_TEETH:
        surface = df[f"O_{tooth}B"]
        present.append(surface.isin(PRESENT_CODES).astype(float))
        missing.append(surface.isin(MISSING_CODES).astype(float))
        unknown.append(surface.isin(UNKNOWN_CODES).astype(float) + surface.isna().astype(float))

    df["n_teeth"] = pd.concat(present, axis=1).sum(axis=1)
    df["n_missing_teeth"] = pd.concat(missing, axis=1).sum(axis=1)
    df["n_unknown_surfaces"] = pd.concat(unknown, axis=1).sum(axis=1)
    df["exam"] = (df.n_teeth + df.n_missing_teeth) > 0

    adults = df[df.age >= 19].copy()
    adults["age2"] = adults.age ** 2

    print(f"adults aged 19 and over          : {len(adults)}")
    print(f"  received the oral examination  : {int(adults.exam.sum())}")
    print(f"  household income recorded      : {int(adults.ho_incm5.notna().sum())}")
    print(f"  aged 30+ with attainment       : {int(adults.loc[adults.age >= 30, 'edu'].notna().sum())}")
    print(f"mean unrecordable surfaces       : {adults.n_unknown_surfaces.mean():.2f}")

    adults.to_pickle(OUT / "01_sample.pkl")


if __name__ == "__main__":
    main()

"""Verify the Newton-Raphson fit against the statsmodels binomial GLM with frequency weights.

Run with: python tests/test_against_statsmodels.py
The test uses simulated data, so it does not require the survey file.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inequality import weighted_logit  # noqa: E402


def main() -> None:
    import statsmodels.api as sm

    rng = np.random.default_rng(20260921)
    n = 5000
    rank = rng.uniform(0, 1, n)
    age = rng.normal(55, 15, n)
    sex = rng.integers(0, 2, n).astype(float)
    weight = rng.gamma(shape=4, scale=0.5, size=n)
    linear = -1.2 + 1.1 * rank + 0.02 * (age - 55) + 0.3 * sex
    y = rng.binomial(1, 1 / (1 + np.exp(-linear))).astype(float)

    X = np.column_stack([np.ones(n), rank, age, age ** 2, sex])
    ours = weighted_logit(X, y, weight)
    theirs = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weight).fit().params

    difference = np.max(np.abs(ours - theirs))
    print("maximum absolute difference in coefficients:", f"{difference:.2e}")
    assert difference < 1e-4, "fits disagree beyond four decimal places"
    print("PASS: the two fits agree to at least four decimal places")


if __name__ == "__main__":
    main()

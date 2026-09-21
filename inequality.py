"""Estimation engine: weighted logistic fit, fractional rank, three estimands, agreement.

All estimands come from one fitted model per indicator, so that differences between them
reflect the contrast taken rather than a difference in specification.
"""
from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

MEASURES = ["conditional_or", "prevalence_ratio", "slope_index", "unadjusted_pred_ratio",
            "marginal_or", "pred_at_means", "noncollapsibility", "scale_component"]


def expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def weighted_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                   max_iter: int = 80, tol: float = 1e-9) -> np.ndarray | None:
    """Newton-Raphson fit of a survey-weighted logistic regression."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-10, 1 - 1e-10)
        gradient = X.T @ (w * (y - p))
        weights = w * p * (1 - p)
        hessian = X.T @ (X * weights[:, None])
        try:
            step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-9, gradient)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return beta
    return None


def fractional_rank(code: np.ndarray, w: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Weighted fractional (ridit) rank: the midpoint of each category's cumulative share.

    `levels` runs from the most advantaged to the most disadvantaged category, so that the
    rank is 0 at the most advantaged and 1 at the most disadvantaged position.
    """
    totals = np.array([w[code == level].sum() for level in levels], dtype=float)
    share = totals / totals.sum()
    midpoints = np.cumsum(share) - share / 2
    rank = np.empty(len(code))
    for level, midpoint in zip(levels, midpoints):
        rank[code == level] = midpoint
    return rank


def estimands(y: np.ndarray, w: np.ndarray, rank: np.ndarray,
              covariates: Sequence[np.ndarray], quadratic_rank: bool = False) -> dict | None:
    """Three relative estimands, the slope index and the decomposition of their difference."""
    columns = [np.ones(len(y)), rank]
    if quadratic_rank:
        columns.append(rank ** 2)
    X = np.column_stack(columns + list(covariates))
    beta = weighted_logit(X, y, w)
    if beta is None:
        return None

    # standardised prevalences with the rank set to 1 and to 0, averaged with the weights
    X1, X0 = X.copy(), X.copy()
    X1[:, 1] = 1.0
    X0[:, 1] = 0.0
    if quadratic_rank:
        X1[:, 2] = 1.0
        X0[:, 2] = 0.0
    p1 = np.average(expit(X1 @ beta), weights=w)
    p0 = np.average(expit(X0 @ beta), weights=w)

    rank_effect = beta[1] + (beta[2] if quadratic_rank else 0.0)
    conditional_or = np.exp(rank_effect)
    marginal_or = (p1 / (1 - p1)) / (p0 / (1 - p0))
    prevalence_ratio = p1 / p0

    # variant reported in some applied work: predicted probabilities at covariate means
    means = [np.average(c, weights=w) for c in covariates]
    offset = beta[0] + sum(b * m for b, m in zip(beta[(3 if quadratic_rank else 2):], means))
    pred_at_means = expit(offset + rank_effect) / expit(offset)

    # unadjusted predicted-probability ratio, the default of inequality-monitoring software
    Xu = np.column_stack([np.ones(len(y)), rank])
    beta_u = weighted_logit(Xu, y, w)
    if beta_u is None:
        return None
    unadjusted = expit(beta_u[0] + beta_u[1]) / expit(beta_u[0])

    return {
        "conditional_or": conditional_or,
        "marginal_or": marginal_or,
        "prevalence_ratio": prevalence_ratio,
        "slope_index": 100 * (p1 - p0),
        "pred_at_means": pred_at_means,
        "unadjusted_pred_ratio": unadjusted,
        "noncollapsibility": conditional_or / marginal_or,
        "scale_component": marginal_or / prevalence_ratio,
        "prevalence": 100 * np.average(y, weights=w),
        "p_advantaged": 100 * p0,
        "p_disadvantaged": 100 * p1,
        "n": len(y),
    }


def magnitude(values: np.ndarray, ratio: bool = True) -> np.ndarray:
    """Magnitude of inequality irrespective of direction.

    Relative estimands are ranked by the absolute logarithm, the slope index by its absolute
    value. The rule matters because one indicator (any caries experience) has a relative
    estimand below unity.
    """
    values = np.asarray(values, dtype=float)
    return np.abs(np.log(np.clip(values, 1e-9, None))) if ratio else np.abs(values)


def _ranks(matrix: np.ndarray) -> np.ndarray:
    order = (-matrix).argsort(axis=1)
    out = np.zeros_like(matrix)
    for row in range(matrix.shape[0]):
        for position, column in enumerate(order[row]):
            out[row, column] = position + 1
    return out


def agreement(point: dict, draws: dict, indicators: Sequence[str],
              measure_a: str, measure_b: str) -> dict:
    """Spearman correlation, discordant pairs and rank shifts between two rankings."""
    ia, ib = MEASURES.index(measure_a), MEASURES.index(measure_b)
    ratio_a = measure_a != "slope_index"
    ratio_b = measure_b != "slope_index"

    a = magnitude([point[i][measure_a] for i in indicators], ratio_a)
    b = magnitude([point[i][measure_b] for i in indicators], ratio_b)
    pairs = list(itertools.combinations(range(len(indicators)), 2))
    discordant = sum(1 for i, j in pairs if np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]))

    M = np.stack([draws[i] for i in indicators], axis=1)
    A = magnitude(M[:, :, ia], ratio_a)
    B = magnitude(M[:, :, ib], ratio_b)
    rho_boot = np.array([spearmanr(A[t], B[t]).statistic for t in range(A.shape[0])])
    RA, RB = _ranks(A), _ranks(B)
    shift = {indicators[k]: {
        "median_rank_a": float(np.median(RA[:, k])),
        "median_rank_b": float(np.median(RB[:, k])),
        "p_shift_ge_2": float(100 * np.mean(np.abs(RA[:, k] - RB[:, k]) >= 2)),
    } for k in range(len(indicators))}

    return {
        "rho": float(spearmanr(a, b).statistic),
        "rho_ci": [float(np.percentile(rho_boot, 2.5)), float(np.percentile(rho_boot, 97.5))],
        "discordant_pairs": int(discordant),
        "n_pairs": len(pairs),
        "pct_replicates_imperfect": float(100 * np.mean(rho_boot < 0.999)),
        "rank_shift": shift,
    }


def pairwise_reversal(draws: dict, indicators: Sequence[str],
                      measure_a: str = "conditional_or",
                      measure_b: str = "prevalence_ratio") -> list[dict]:
    """Proportion of replicates in which each pair is ordered differently by two measures."""
    ia, ib = MEASURES.index(measure_a), MEASURES.index(measure_b)
    out = []
    for i, j in itertools.combinations(range(len(indicators)), 2):
        a_i, a_j = magnitude(draws[indicators[i]][:, ia]), magnitude(draws[indicators[j]][:, ia])
        b_i, b_j = magnitude(draws[indicators[i]][:, ib]), magnitude(draws[indicators[j]][:, ib])
        differs = np.sign(a_i - a_j) != np.sign(b_i - b_j)
        out.append({"indicator_a": indicators[i], "indicator_b": indicators[j],
                    "p_ordered_differently": float(differs.mean())})
    return sorted(out, key=lambda r: -r["p_ordered_differently"])

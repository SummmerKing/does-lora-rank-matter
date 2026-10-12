"""pass@1 with bootstrap CIs, paired comparisons and Holm correction.

Results for one config are a (seeds x problems) 0/1 matrix. CIs resample
problems; comparisons use a sign-flip permutation test on the per-problem
differences.
"""

from __future__ import annotations

from math import comb

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021) from n samples with c
    correct: 1 - C(n-c, k) / C(n, k), computed as a stable product."""
    if not 0 <= c <= n or k < 1 or k > n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _as_matrix(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def summarize(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    x = _as_matrix(x)
    per_problem = x.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_problem.size, size=(n_boot, per_problem.size))
    boots = per_problem[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    seed_means = x.mean(axis=1)
    return {
        "pass@1": float(per_problem.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "seed_sd": float(seed_means.std(ddof=1)) if x.shape[0] > 1 else 0.0,
        "n_seeds": int(x.shape[0]),
        "n_problems": int(x.shape[1]),
    }


def paired_comparison(a, b, n_perm: int = 20_000, n_boot: int = 10_000, alpha: float = 0.05,
                      seed: int = 0) -> dict:
    """Compare configuration A against B on the same problems."""
    a, b = _as_matrix(a), _as_matrix(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("configurations must be evaluated on the same problems")
    d = a.mean(axis=0) - b.mean(axis=0)
    observed = d.mean()
    rng = np.random.default_rng(seed)

    nonzero = d[d != 0]
    if nonzero.size == 0:
        p_value = 1.0
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, nonzero.size))
        null = (signs * nonzero).sum(axis=1) / d.size
        # +1 correction keeps the Monte Carlo p-value valid (never exactly 0).
        p_value = (1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (n_perm + 1)

    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(observed), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p_value)}


def mcnemar_exact(a, b) -> dict:
    """Exact McNemar test for two 0/1 vectors on the same problems."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()

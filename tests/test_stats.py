import itertools
from math import comb

import numpy as np
import pytest

from lora_code.stats import holm, mcnemar_exact, pass_at_k, paired_comparison, summarize


@pytest.mark.parametrize("n,c,k", [(10, 3, 1), (10, 3, 5), (20, 0, 5), (20, 20, 1), (7, 2, 6), (5, 1, 5)])
def test_pass_at_k_matches_combinatorial_definition(n, c, k):
    assert pass_at_k(n, c, k) == pytest.approx(1 - comb(n - c, k) / comb(n, k))


def test_pass_at_k_is_unbiased_by_enumeration():
    """E over all size-k draws from n samples of [any correct] equals pass_at_k."""
    n, c, k = 8, 3, 3
    samples = [1] * c + [0] * (n - c)
    draws = list(itertools.combinations(range(n), k))
    empirical = np.mean([any(samples[i] for i in d) for d in draws])
    assert pass_at_k(n, c, k) == pytest.approx(empirical)


def test_pass_at_1_is_fraction_correct():
    assert pass_at_k(10, 4, 1) == pytest.approx(0.4)


def test_summarize_ci_covers_mean_and_seed_sd():
    rng = np.random.default_rng(0)
    x = (rng.random((3, 200)) < 0.4).astype(int)
    s = summarize(x)
    assert s["ci_low"] < s["pass@1"] < s["ci_high"]
    assert s["seed_sd"] == pytest.approx(x.mean(axis=1).std(ddof=1))
    assert (s["n_seeds"], s["n_problems"]) == (3, 200)


def test_bootstrap_ci_has_roughly_nominal_coverage():
    rng = np.random.default_rng(1)
    p, covered, trials = 0.3, 0, 200
    for t in range(trials):
        s = summarize((rng.random(150) < p).astype(int), n_boot=2000, seed=t)
        covered += s["ci_low"] <= p <= s["ci_high"]
    assert 0.88 <= covered / trials <= 0.99


def test_paired_comparison_detects_real_difference_and_not_null():
    rng = np.random.default_rng(2)
    base = (rng.random(300) < 0.4).astype(int)
    better = base.copy()
    better[np.flatnonzero(base == 0)[:45]] = 1  # +15 points on the same problems
    r = paired_comparison(better, base, seed=0)
    assert r["diff"] == pytest.approx(0.15)
    assert r["p_value"] < 0.001 and r["ci_low"] > 0

    same = paired_comparison(base, base)
    assert same["diff"] == 0 and same["p_value"] == 1.0


def test_paired_comparison_false_positive_rate():
    rng = np.random.default_rng(3)
    rejections = 0
    for t in range(200):
        a = (rng.random(150) < 0.5).astype(int)
        b = (rng.random(150) < 0.5).astype(int)
        rejections += paired_comparison(a, b, n_perm=1000, n_boot=200, seed=t)["p_value"] < 0.05
    assert rejections / 200 <= 0.09


def test_mcnemar_exact():
    a = np.array([1] * 10 + [0] * 2 + [1] * 50 + [0] * 38)
    b = np.array([0] * 10 + [1] * 2 + [1] * 50 + [0] * 38)
    r = mcnemar_exact(a, b)
    assert (r["only_a"], r["only_b"]) == (10, 2)
    expected = 2 * sum(comb(12, i) for i in range(3)) / 2 ** 12
    assert r["p_value"] == pytest.approx(expected)
    try:
        from scipy.stats import binomtest
        assert r["p_value"] == pytest.approx(binomtest(2, 12, 0.5).pvalue)
    except ImportError:
        pass


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])

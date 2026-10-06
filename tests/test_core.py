"""Synthetic-data tests for core math (no network)."""

import math

import pytest

from pm_mispricing.core import (
    bayesian_estimate,
    beta_posterior_mean,
    beta_posterior_std,
    clip_prob,
    flag_signal,
    implied_prob_from_price,
    mispricing_edge,
    normalize_prices,
    overround,
    screen,
)


def test_implied_prob_clips_to_unit_interval():
    assert implied_prob_from_price(0.62) == pytest.approx(0.62)
    assert implied_prob_from_price(-0.5) == 0.0
    assert implied_prob_from_price(1.7) == 1.0
    assert clip_prob(0.0) == 0.0 and clip_prob(1.0) == 1.0


def test_normalize_prices_sums_to_one():
    probs = normalize_prices([0.60, 0.55])  # 15% overround book
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.60 / 1.15)


def test_normalize_rejects_empty_book():
    with pytest.raises(ValueError):
        normalize_prices([0.0, 0.0])


def test_overround_positive_for_vig_book():
    assert overround([0.60, 0.55]) == pytest.approx(0.15)
    assert overround([0.5, 0.5]) == pytest.approx(0.0)


def test_beta_posterior_mean_formula():
    # Prior Beta(2, 2) + 7 wins in 10 trials -> (2+7)/(4+10) = 9/14
    assert beta_posterior_mean(2.0, 2.0, 7, 10) == pytest.approx(9 / 14)


def test_beta_posterior_no_data_reduces_to_prior():
    assert beta_posterior_mean(3.0, 7.0, 0, 0) == pytest.approx(0.3)
    mean, std = bayesian_estimate(3.0, 7.0, 0, 0)
    assert mean == pytest.approx(0.3)
    assert std > 0


def test_beta_posterior_std_shrinks_with_data():
    assert beta_posterior_std(2.0, 2.0, 50, 100) < beta_posterior_std(2.0, 2.0, 5, 10)


def test_beta_posterior_rejects_bad_inputs():
    with pytest.raises(ValueError):
        beta_posterior_mean(0.0, 2.0, 1, 2)
    with pytest.raises(ValueError):
        beta_posterior_mean(2.0, 2.0, 5, 3)  # wins > trials


def test_edge_sign_convention():
    # Model above market -> underpriced Yes -> positive edge
    assert mispricing_edge(0.40, 0.55) == pytest.approx(0.15)
    assert mispricing_edge(0.70, 0.55) == pytest.approx(-0.15)


def test_flag_signal_thresholds():
    assert flag_signal(0.08, threshold=0.05) == "BUY_YES"
    assert flag_signal(-0.08, threshold=0.05) == "BUY_NO"
    assert flag_signal(0.03, threshold=0.05) == "FAIR"
    assert flag_signal(0.05, threshold=0.05) == "FAIR"  # boundary is fair
    with pytest.raises(ValueError):
        flag_signal(0.1, threshold=-0.01)


def test_screen_flags_known_mispricing_and_sorts():
    records = [
        {"question": "fair coin", "market_prob": 0.50, "model_prob": 0.51},
        {"question": "underpriced", "market_prob": 0.30, "model_prob": 0.55},
        {"question": "overpriced", "market_prob": 0.80, "model_prob": 0.55},
    ]
    df = screen(records, threshold=0.05)
    assert list(df["signal"]) == ["BUY_YES", "BUY_NO", "FAIR"]
    assert df["abs_edge"].is_monotonic_decreasing
    assert df.iloc[0]["question"] == "underpriced"


def test_screen_derives_bayesian_baseline_from_counts():
    records = [
        {
            "question": "hot team",
            "market_prob": 0.40,
            "prior_alpha": 2.0,
            "prior_beta": 2.0,
            "wins": 70,
            "trials": 100,
        }
    ]
    df = screen(records, threshold=0.05)
    expected = (2 + 70) / (4 + 100)
    assert df.iloc[0]["model_prob"] == pytest.approx(expected, abs=1e-4)
    assert df.iloc[0]["signal"] == "BUY_YES"
    assert df.iloc[0]["model_std"] > 0


def test_screen_accepts_explicit_model_prob_over_prior():
    rec = {
        "question": "x",
        "market_prob": 0.5,
        "model_prob": 0.9,
        "prior_alpha": 1.0,
        "prior_beta": 99.0,
        "wins": 0,
        "trials": 0,
    }
    df = screen([rec])
    assert df.iloc[0]["model_prob"] == pytest.approx(0.9)

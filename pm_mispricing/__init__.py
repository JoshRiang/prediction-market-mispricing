"""Prediction market mispricing screener.

Compares market-implied probabilities against a Bayesian baseline
estimate and flags markets where the gap exceeds a threshold.
"""

from .core import (
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
from .io import fetch_polymarket_markets, load_markets, synthetic_markets

__all__ = [
    "bayesian_estimate",
    "beta_posterior_mean",
    "beta_posterior_std",
    "clip_prob",
    "fetch_polymarket_markets",
    "flag_signal",
    "implied_prob_from_price",
    "load_markets",
    "mispricing_edge",
    "normalize_prices",
    "overround",
    "screen",
    "synthetic_markets",
]

__version__ = "0.1.0"

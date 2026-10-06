"""Core math: implied probabilities, Bayesian baseline, mispricing screener."""

from __future__ import annotations

import math

import pandas as pd


def clip_prob(p: float) -> float:
    """Clip a raw price/probability into [0, 1]."""
    return min(1.0, max(0.0, float(p)))


def implied_prob_from_price(price: float) -> float:
    """Convert a binary Yes-token price in dollars to an implied probability.

    Polymarket Yes/No tokens pay out $1 on resolution, so under
    risk-neutrality the price *is* the market's probability estimate.
    """
    return clip_prob(price)


def normalize_prices(prices: list[float]) -> list[float]:
    """Remove book overround: divide each price by the sum so probs add to 1."""
    total = sum(float(p) for p in prices)
    if total <= 0:
        raise ValueError(f"Cannot normalize non-positive price sum: {total}")
    return [clip_prob(p / total) for p in prices]


def overround(prices: list[float]) -> float:
    """Book overround (vig): sum(prices) - 1. Positive = market takes a cut."""
    return float(sum(prices)) - 1.0


def beta_posterior_mean(prior_alpha: float, prior_beta: float, wins: int, trials: int) -> float:
    """Posterior mean of a Beta-Binomial model.

    Prior Beta(alpha, beta) updated with `wins` successes in `trials`
    observations: posterior Beta(alpha + wins, beta + trials - wins).
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior parameters must be positive")
    if trials < 0 or wins < 0 or wins > trials:
        raise ValueError(f"Invalid outcome counts: wins={wins}, trials={trials}")
    return (prior_alpha + wins) / (prior_alpha + prior_beta + trials)


def beta_posterior_std(prior_alpha: float, prior_beta: float, wins: int, trials: int) -> float:
    """Posterior standard deviation of the Beta-Binomial model."""
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior parameters must be positive")
    if trials < 0 or wins < 0 or wins > trials:
        raise ValueError(f"Invalid outcome counts: wins={wins}, trials={trials}")
    a = prior_alpha + wins
    b = prior_beta + trials - wins
    s = a + b
    return math.sqrt(a * b / (s * s * (s + 1.0)))


def bayesian_estimate(
    prior_alpha: float, prior_beta: float, wins: int, trials: int
) -> tuple[float, float]:
    """Bayesian baseline probability estimate: (posterior mean, posterior std)."""
    return (
        beta_posterior_mean(prior_alpha, prior_beta, wins, trials),
        beta_posterior_std(prior_alpha, prior_beta, wins, trials),
    )


def mispricing_edge(market_prob: float, model_prob: float) -> float:
    """Signed edge: positive means the market underprices the outcome."""
    return float(model_prob) - float(market_prob)


def flag_signal(edge: float, threshold: float = 0.05) -> str:
    """Map a signed edge to a trading signal."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if edge > threshold:
        return "BUY_YES"
    if edge < -threshold:
        return "BUY_NO"
    return "FAIR"


def screen(records: list[dict], threshold: float = 0.05) -> pd.DataFrame:
    """Screen a list of market records for mispricing.

    Each record needs: ``question``, ``market_prob``, plus either
    ``model_prob`` directly or ``prior_alpha``/``prior_beta``/``wins``/
    ``trials`` to derive the Bayesian baseline. Returns a DataFrame
    sorted by descending absolute edge.
    """
    rows: list[dict] = []
    for rec in records:
        market_prob = clip_prob(rec["market_prob"])
        if "model_prob" in rec and rec["model_prob"] is not None:
            model_prob = clip_prob(rec["model_prob"])
            model_std = rec.get("model_std")
        else:
            mean, std = bayesian_estimate(
                float(rec.get("prior_alpha", 1.0)),
                float(rec.get("prior_beta", 1.0)),
                int(rec.get("wins", 0)),
                int(rec.get("trials", 0)),
            )
            model_prob, model_std = mean, std
        edge = mispricing_edge(market_prob, model_prob)
        rows.append(
            {
                "question": rec.get("question", "?"),
                "market_prob": round(market_prob, 4),
                "model_prob": round(model_prob, 4),
                "model_std": round(model_std, 4) if model_std is not None else None,
                "edge": round(edge, 4),
                "abs_edge": round(abs(edge), 4),
                "signal": flag_signal(edge, threshold),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=["question", "market_prob", "model_prob", "model_std", "edge", "abs_edge", "signal"],
    )
    return df.sort_values("abs_edge", ascending=False).reset_index(drop=True)

"""Data loading: live Polymarket fetch with offline synthetic fallback."""

from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


def _parse_yes_price(market: dict) -> float | None:
    """Extract the Yes-token price from a Gamma market object."""
    raw = market.get("outcomePrices")
    if raw is None:
        for key in ("lastTradePrice", "bestBid"):
            if market.get(key) is not None:
                try:
                    return float(market[key])
                except (TypeError, ValueError):
                    continue
        return None
    try:
        prices = json.loads(raw) if isinstance(raw, str) else list(raw)
        return float(prices[0])
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        return None


def fetch_polymarket_markets(limit: int = 20, timeout: int = 10) -> list[dict]:
    """Fetch live open markets from the Polymarket Gamma API.

    Returns records with ``question``, ``market_prob``, and a neutral
    ``prior_alpha``/``prior_beta`` (no outcome history is available via
    the public API, so the CLI compares live prices against the
    configurable base-rate prior). Raises on network/API failure so the
    caller can fall back to synthetic data.
    """
    import requests

    resp = requests.get(
        GAMMA_EVENTS_URL,
        params={"limit": limit, "closed": "false", "order": "volume24hr", "ascending": "false"},
        timeout=timeout,
    )
    resp.raise_for_status()
    events = resp.json()
    if isinstance(events, dict):
        events = events.get("events", events.get("data", []))
    records: list[dict] = []
    for event in events:
        for market in event.get("markets", []) or []:
            price = _parse_yes_price(market)
            if price is None:
                continue
            question = market.get("question") or event.get("title", "Unknown market")
            records.append(
                {
                    "question": question,
                    "market_prob": min(1.0, max(0.0, price)),
                    "prior_alpha": 1.0,
                    "prior_beta": 1.0,
                    "wins": 0,
                    "trials": 0,
                    "source": "polymarket-live",
                }
            )
            if len(records) >= limit:
                return records
    if not records:
        raise ValueError("Gamma API returned no parseable markets")
    return records


def synthetic_markets(n: int = 12, seed: int = 0) -> list[dict]:
    """Deterministic synthetic markets with known mispricing.

    Each market has a hidden true probability; the observed market price
    is noisy and a few markets are deliberately mispriced, while
    wins/trials are drawn from the true probability so the Bayesian
    baseline can recover it.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    true_ps = rng.uniform(0.15, 0.85, size=n)
    for i, true_p in enumerate(true_ps):
        noise = rng.normal(0.0, 0.04)
        bias = 0.0
        if i % 4 == 0:
            bias = 0.12  # overpriced Yes
        elif i % 4 == 1:
            bias = -0.12  # underpriced Yes
        market_prob = min(0.98, max(0.02, float(true_p + noise + bias)))
        trials = int(rng.integers(30, 80))
        wins = int(rng.binomial(trials, float(true_p)))
        records.append(
            {
                "question": f"Synthetic market {i + 1} (true p={true_p:.2f})",
                "market_prob": market_prob,
                "prior_alpha": 2.0,
                "prior_beta": 2.0,
                "wins": wins,
                "trials": trials,
                "source": "synthetic",
            }
        )
    return records


def load_markets(limit: int = 20, use_live: bool = True) -> tuple[list[dict], str]:
    """Load markets, preferring live data with synthetic fallback.

    Returns (records, source) where source is 'polymarket-live' or 'synthetic'.
    """
    if use_live:
        try:
            records = fetch_polymarket_markets(limit=limit)
            return records, "polymarket-live"
        except Exception as exc:  # network/API failure -> offline fallback
            logger.warning("Live fetch failed (%s); using synthetic fallback.", exc)
    return synthetic_markets(n=limit), "synthetic"

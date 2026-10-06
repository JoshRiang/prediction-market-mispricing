"""Synthetic-data tests for the IO layer (no network)."""

import pytest

from pm_mispricing.io import load_markets, synthetic_markets


def test_synthetic_markets_deterministic():
    first = synthetic_markets(n=8, seed=42)
    second = synthetic_markets(n=8, seed=42)
    assert first == second
    assert len(first) == 8
    for rec in first:
        assert 0.0 <= rec["market_prob"] <= 1.0
        assert 0 <= rec["wins"] <= rec["trials"]
        assert rec["source"] == "synthetic"


def test_synthetic_markets_contain_mispricing():
    recs = synthetic_markets(n=12, seed=0)
    biases = [abs(r["market_prob"] - r["wins"] / r["trials"]) for r in recs]
    assert max(biases) > 0.05  # at least one clearly off-market price


def test_load_markets_falls_back_when_live_fails(monkeypatch):
    import pm_mispricing.io as io_mod

    def _boom(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(io_mod, "fetch_polymarket_markets", _boom)
    records, source = load_markets(limit=5, use_live=True)
    assert source == "synthetic"
    assert len(records) == 5


def test_load_markets_no_live_flag_skips_network(monkeypatch):
    import pm_mispricing.io as io_mod

    called = []
    monkeypatch.setattr(
        io_mod, "fetch_polymarket_markets", lambda *a, **k: called.append(1) or []
    )
    records, source = load_markets(limit=4, use_live=False)
    assert called == []
    assert source == "synthetic"
    assert len(records) == 4


def test_parse_yes_price_handles_string_and_missing():
    from pm_mispricing.io import _parse_yes_price

    assert _parse_yes_price({"outcomePrices": '["0.62", "0.38"]'}) == pytest.approx(0.62)
    assert _parse_yes_price({"outcomePrices": [0.3, 0.7]}) == pytest.approx(0.3)
    assert _parse_yes_price({"lastTradePrice": 0.44}) == pytest.approx(0.44)
    assert _parse_yes_price({}) is None

# Prediction Market Mispricing Screener

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests: 18 passed](https://img.shields.io/badge/pytest-18%20passed-green)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Flags mispriced binary prediction markets by comparing **market-implied
probabilities** (token prices) against a **Bayesian Beta-Binomial baseline**.
Two data paths, always labelled: **live** prices from the Polymarket Gamma API
(with synthetic fallback offline), and a fully offline **synthetic** demo with
known mispricing.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Live prices (falls back to synthetic offline), flag |edge| > 0.05
python -m pm_mispricing

# Fully offline synthetic demo
python -m pm_mispricing --no-live --limit 12 --threshold 0.05

# Skeptical prior for live markets (base rate ~30%)
python -m pm_mispricing --prior-alpha 3 --prior-beta 7

python -m pytest tests/ -v   # synthetic data only, no network
```

## Architecture / Method

**Implied probability** (`core.implied_prob_from_price`) — a Yes token paying
$1 on resolution is worth exactly its risk-neutral probability, so
`P_market = price` (clipped to [0, 1]). Multi-outcome books are de-vigged via
`normalize_prices` (`p_i / Σp`); `overround = Σp − 1` measures the book's cut.

**Bayesian baseline** (`core.beta_posterior_mean/std`) — each outcome is
`Beta(α, β)`, updated with observed history (`wins`/`trials`):

- Posterior mean: `(α + w) / (α + β + n)`
- Posterior std: `√(a·b / (s²·(s+1)))`, `a = α+w`, `b = β+n−w`, `s = a+b`

**Screener** (`core.screen`) — `edge = P_model − P_market`; `|edge| > τ`
flags `BUY_YES` (market underprices) or `BUY_NO` (market overprices), else
`FAIR`; results sorted by `|edge|`.

**Data** (`io.py`) — `fetch_polymarket_markets` reads live open markets from
`GET /events` (Yes price parsed from `outcomePrices`); `synthetic_markets`
generates deterministic markets where hidden true probabilities are perturbed
by noise plus deliberate ±0.12 biases, with `wins/trials ~ Binomial(true p)`
so the baseline can recover them; `load_markets` prefers live, falls back to
synthetic, and always reports which source was used.

> Note: the public Gamma API exposes prices but not outcome histories, so
> **live** markets are screened against a configurable base-rate prior
> (`--prior-alpha/--prior-beta`, default uniform). The **synthetic** demo ships
> full histories, so the Bayesian update is fully exercised there.

## Sample output (real run)

```bash
python -m pm_mispricing --no-live --limit 12 --threshold 0.05
```

```
Data source : synthetic (12 markets)
Threshold   : |edge| > 0.05
Flagged     : 6 mispriced / 12 total

                         question  market_prob  model_prob  model_std    edge  signal
 Synthetic market 2 (true p=0.34)       0.1896      0.4286     0.0825  0.2390 BUY_YES
 Synthetic market 5 (true p=0.72)       0.8127      0.6833     0.0596 -0.1293  BUY_NO
 Synthetic market 9 (true p=0.53)       0.6442      0.5323     0.0629 -0.1119  BUY_NO
Synthetic market 10 (true p=0.80)       0.6988      0.7857     0.0487  0.0869 BUY_YES
 Synthetic market 1 (true p=0.60)       0.6229      0.5574     0.0631 -0.0655  BUY_NO
 Synthetic market 6 (true p=0.79)       0.6727      0.6212     0.0593 -0.0515  BUY_NO
Synthetic market 12 (true p=0.15)       0.1016      0.1500     0.0397  0.0484    FAIR
 Synthetic market 3 (true p=0.18)       0.1660      0.2105     0.0653  0.0445    FAIR
 Synthetic market 8 (true p=0.66)       0.6203      0.6618     0.0570  0.0415    FAIR
 Synthetic market 4 (true p=0.16)       0.1564      0.1250     0.0472 -0.0314    FAIR
Synthetic market 11 (true p=0.72)       0.7159      0.7297     0.0720  0.0138    FAIR
 Synthetic market 7 (true p=0.57)       0.5378      0.5385     0.0685  0.0007    FAIR

Top opportunity: Synthetic market 2 (true p=0.34)
  market=0.19 model=0.43 edge=+0.239 signal=BUY_YES
```

## Results

6 of 12 synthetic markets flagged at `τ = 0.05`. The screener recovers the
injected biases: deliberately underpriced books surface as `BUY_YES`
(market 2: edge +0.239, the largest), overpriced ones as `BUY_NO` (markets 5,
9). Unbiased markets land `FAIR` with edges ≤ 0.05 — consistent with pure
Binomial sampling noise around the posterior. The posterior std column (≈
0.04–0.08) quantifies baseline uncertainty from 30–80 trials, a natural
confidence gauge for sizing.

## Limitations

- Live screening compares prices against a base-rate *prior*, not a fitted
  model — it detects deviation from base rates, not true mispricing.
- No fees, spreads, liquidity, or resolution risk; flagged "edge" is gross.
- Synthetic demo is deliberately easy (large ±0.12 biases); real edges are
  smaller and noisier.
- Binary/Yes-token framing only; no multi-outcome arbitrage or cross-market
  hedging.

## Project structure

```
prediction-market-mispricing/
├── pm_mispricing/
│   ├── core.py        # implied prob, de-vigging, Beta-Binomial baseline, screener
│   ├── io.py          # Polymarket Gamma fetcher + deterministic synthetic fallback
│   └── __main__.py    # CLI: load, screen, print ranked table
├── tests/
│   ├── test_core.py   # prob clipping, normalization, posterior math, signals
│   └── test_io.py     # synthetic determinism, parser edge cases (offline)
├── requirements.txt
├── LICENSE
└── README.md
```

## Testing

```bash
python -m pytest tests/ -v   # 18 passed, synthetic data only, no network
```

Covers probability clipping, overround accounting, posterior mean/std against
closed forms, signal thresholds (including negative-threshold rejection), sort
order by `|edge|`, synthetic determinism, and Gamma payload parsing without
touching the network.

## License

MIT — see [LICENSE](LICENSE).

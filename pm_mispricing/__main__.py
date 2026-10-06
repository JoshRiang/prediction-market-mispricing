"""CLI demo: python -m pm_mispricing [--no-live] [--limit N] [--threshold T]."""

from __future__ import annotations

import argparse

from .core import screen
from .io import load_markets


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prediction market mispricing screener")
    p.add_argument("--limit", type=int, default=12, help="Max markets to screen")
    p.add_argument("--threshold", type=float, default=0.05, help="Min |edge| to flag")
    p.add_argument("--no-live", action="store_true", help="Skip live fetch, use synthetic data")
    p.add_argument(
        "--prior-alpha", type=float, default=1.0, help="Base-rate prior alpha for live markets"
    )
    p.add_argument(
        "--prior-beta", type=float, default=1.0, help="Base-rate prior beta for live markets"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records, source = load_markets(limit=args.limit, use_live=not args.no_live)
    if source == "polymarket-live":
        for rec in records:  # live prices vs configurable base-rate prior
            rec["prior_alpha"] = args.prior_alpha
            rec["prior_beta"] = args.prior_beta
    df = screen(records, threshold=args.threshold)
    print(f"Data source : {source} ({len(df)} markets)")
    print(f"Threshold   : |edge| > {args.threshold:.2f}")
    flagged = df[df["signal"] != "FAIR"]
    print(f"Flagged     : {len(flagged)} mispriced / {len(df)} total")
    print()
    print(df.drop(columns=["abs_edge"]).to_string(index=False))
    if not flagged.empty:
        print()
        print("Top opportunity:", flagged.iloc[0]["question"])
        print(
            f"  market={flagged.iloc[0]['market_prob']:.2f} "
            f"model={flagged.iloc[0]['model_prob']:.2f} "
            f"edge={flagged.iloc[0]['edge']:+.3f} "
            f"signal={flagged.iloc[0]['signal']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

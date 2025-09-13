"""
Regime-Switching Portfolio (BTC, VOO, VUG, GLDM) using Polygon.io REST data

Quick start:
  pip install pandas numpy matplotlib hmmlearn pydantic polygon-api-client

Run:
  python regime_switching_portfolio_polygon.py --save signals.csv
  # (The --save flag is optional; if provided, writes Date/Regime/Weights to CSV)

Notes:
- Set your REST API key via env var POLYGON_API_KEY (recommended), or edit API_KEY_FALLBACK below.
- If Polygon returns zero bars, you are likely using the wrong credential (e.g., Flat Files keys).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from pydantic import BaseModel
from polygon import RESTClient


# ========== USER CONFIG ==========
# Prefer env var, fallback to a literal string if you insist (not recommended for security).
API_KEY_FALLBACK = ""  # e.g., "your-rest-api-key-here"
API_KEY = os.getenv("POLYGON_API_KEY", API_KEY_FALLBACK)

ASSETS = ["BTC-USD", "VOO", "VUG", "GLDM", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2025-09-13"
INTERVAL = "day"  # Polygon timespan; can also be "hour", "minute"
# =================================


# ---------------- Data Provider ----------------
class PolygonProvider:
    """Fetch adjusted aggregates from Polygon and return wide Close price DataFrame."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "No API key provided. Set POLYGON_API_KEY env var or fill API_KEY_FALLBACK."
            )
        self.client = RESTClient(api_key)

    @staticmethod
    def _map_symbol(ticker: str) -> str:
        # Polygon uses "X:BTCUSD" for spot BTCUSD; ETFs keep plain symbols.
        return "X:BTCUSD" if ticker.upper() == "BTC-USD" else ticker.upper()

    def get_history(self, tickers: List[str], start: str, end: str, interval: str = "day") -> pd.DataFrame:
        frames = []
        for t in tickers:
            sym = self._map_symbol(t)
            try:
                bars_iter = self.client.list_aggs(
                    ticker=sym,
                    multiplier=1,
                    timespan=interval,
                    from_=start,
                    to=end,
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                )
            except Exception as e:
                raise RuntimeError(f"Polygon request failed for {sym}: {e}")

            rows = []
            for bar in bars_iter:
                ts = getattr(bar, "timestamp", None)
                close = getattr(bar, "close", None)
                if ts is None or close is None:
                    continue
                idx = pd.to_datetime(ts, unit="ms", utc=True).tz_convert("America/New_York").normalize()
                rows.append((idx, float(close)))

            df = pd.DataFrame(rows, columns=["Date", t]).set_index("Date")
            print(f"Fetched {len(df):>5} daily bars for {t} ({sym})")
            frames.append(df)

        if not frames:
            raise ValueError("No data frames returned from Polygon (all tickers empty).")
        out = pd.concat(frames, axis=1).sort_index()
        out = out.dropna(how="all")
        return out


# ---------------- Feature Engineering ----------------
def make_features(prices: pd.Series) -> pd.DataFrame:
    r1 = prices.pct_change()
    vol20 = r1.rolling(20).std()
    vol60 = r1.rolling(60).std()
    mom20 = prices.pct_change(20)
    dd = prices / prices.cummax() - 1
    feats = pd.concat([r1, vol20, vol60, mom20, dd], axis=1)
    feats.columns = ["r1", "vol20", "vol60", "mom20", "dd"]
    return feats.dropna()


# ---------------- Regime Model ----------------
class RegimeModel(BaseModel):
    n_states: int = 3
    window: int = 750  # target window; adapts based on available data

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        Xv = X.values
        dates = X.index
        n = len(Xv)
        if n <= 120:
            raise ValueError(f"Not enough feature rows ({n}). Need >120. Try widening date range.")

        # Leave >= ~60 obs for inference
        eff_window = min(self.window, max(50, n - 60, 100))
        if eff_window >= n:
            eff_window = max(50, n // 2)

        states = []
        for i in range(eff_window, n):
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(Xv[i - eff_window: i])
            _, post = model.score_samples(Xv[i - eff_window: i])
            states.append(int(np.argmax(post[-1])))

        if not states:
            raise ValueError("No states inferred after adapting window.")
        return pd.Series(states, index=dates[eff_window:], name="state")


# ---------------- Regime Naming ----------------
def name_states_by_forward_return(states: pd.Series, r1: pd.Series) -> pd.Series:
    """
    Name regimes by forward 1-week return. Robust if fewer than 3 states observed.
    Best -> BULL; worst -> BEAR; remaining -> HIGHVOL.
    """
    if states.empty:
        raise ValueError("No states inferred — check data length vs window")
    fwd = r1.shift(-5).reindex(states.index)
    uniq = list(np.unique(states))
    means = {s: fwd[states == s].mean() for s in uniq}
    ranked = sorted(uniq, key=lambda s: (means[s] if pd.notna(means[s]) else -np.inf), reverse=True)

    labels = ["BULL", "HIGHVOL", "BEAR"]
    mapping = {}
    for i, s in enumerate(ranked):
        mapping[s] = labels[i] if i < len(labels) else f"STATE_{i}"
    if len(ranked) == 2:
        mapping[ranked[1]] = "BEAR"  # ensure worst is BEAR
    return states.map(mapping)


# ---------------- Portfolio Policy ----------------
POLICY: Dict[str, Dict[str, float]] = {
    "BULL":    {"BTC-USD": 0.30, "VOO": 0.40, "VUG": 0.20, "GLDM": 0.10},
    "HIGHVOL": {"BTC-USD": 0.15, "VOO": 0.30, "VUG": 0.15, "GLDM": 0.40},
    "BEAR":    {"BTC-USD": 0.05, "VOO": 0.20, "VUG": 0.05, "GLDM": 0.70},
}


# ---------------- Backtest & Stats ----------------
def backtest(px: pd.DataFrame, regimes: pd.Series, policy: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Return daily returns DataFrame with columns: strategy, benchmark(SPY)."""
    # Compute returns and align to regime dates
    ret = px.pct_change().reindex(regimes.index).dropna(how="all")
    if ret.empty:
        return pd.DataFrame(columns=["strategy", "benchmark"])

    # Build weights only over strategy assets (ignore SPY for allocation)
    # Aggregate all keys across regimes in POLICY
    all_keys = set()
    for d in policy.values():
        all_keys.update(d.keys())
    # Keep only assets that exist in px and are not SPY
    strat_assets = [a for a in px.columns if a in all_keys and a != "SPY"]
    if not strat_assets:
        # Fallback: use all columns except SPY
        strat_assets = [c for c in px.columns if c != "SPY"]

    # Expand regime->weights to a DataFrame (may miss some columns if a regime lacks a key)
    W = regimes.map(policy).apply(pd.Series)
    # Ensure we have the exact columns and order
    for a in strat_assets:
        if a not in W.columns:
            W[a] = 0.0
    W = W[strat_assets]

    # Trade next day to avoid lookahead and forward-fill gaps
    W = W.shift(1).reindex(ret.index).ffill().fillna(0.0)

    # Strategy daily return
    port = (W[strat_assets] * ret[strat_assets]).sum(axis=1)

    # Benchmark = SPY daily return
    if "SPY" not in ret.columns:
        raise ValueError("SPY data not available for benchmark. Add 'SPY' to ASSETS.")
    bench = ret["SPY"]

    out = pd.DataFrame({"strategy": port, "benchmark": bench}).dropna(how="all")
    return out



def perf_stats(ser: pd.Series) -> pd.Series:
    ser = ser.dropna()
    if ser.empty or ser.std() == 0:
        return pd.Series({"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})
    c = (1 + ser).cumprod()
    cagr = c.iloc[-1] ** (252 / len(c)) - 1
    vol = ser.std() * np.sqrt(252)
    dd = (c / c.cummax() - 1).min()
    sharpe = ser.mean() / ser.std() * np.sqrt(252)
    return pd.Series({"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": dd})


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None, help="Optional CSV filename to save signals")
    args = parser.parse_args()

    provider = PolygonProvider(API_KEY)
    print(f"Downloading Polygon data for {ASSETS} from {START_DATE} to {END_DATE}...")
    px = provider.get_history(ASSETS, START_DATE, END_DATE, interval=INTERVAL)
    if px.empty:
        raise RuntimeError(
            "Polygon returned zero rows across all tickers. "
            "Likely an invalid REST key or insufficient dataset access."
        )
    px = px.dropna(how="all")
    print(f"Downloaded price shape: {px.shape}, dates: {px.index.min().date()} → {px.index.max().date()}")

    # Proxy for regime detection (prefer VOO; fallback to blended VOO/VUG)
    proxy = px["VOO"].dropna()
    if len(proxy) < 180:
        proxy = px[["VOO", "VUG"]].mean(axis=1).dropna()

    feat = make_features(proxy)
    print(f"Feature rows available: {len(feat)}")
    model = RegimeModel()
    states = model.fit_predict(feat)
    print(f"Inferred regime days: {len(states)} ({states.index.min().date()} → {states.index.max().date()})")
    named = name_states_by_forward_return(states, proxy.pct_change())

    # Backtest
    perf = backtest(px, named, POLICY)

    # Save signals if requested
    if args.save:
        signals = pd.DataFrame(index=named.index)
        signals["Regime"] = named
        weights = named.map(POLICY).apply(pd.Series)
        signals = pd.concat([signals, weights], axis=1)
        signals.to_csv(args.save)
        print(f"Signals saved to {args.save}")

    # Summary
    stats = pd.concat(
        {
            "Strategy": perf_stats(perf.get("strategy", pd.Series(dtype=float))),
            "Benchmark": perf_stats(perf.get("benchmark", pd.Series(dtype=float))),
        },
        axis=1,
    ).round(3)
    print("\nPerformance Summary:\n", stats)

    # Plot if we have data
    if not perf.empty and perf.dropna(how="all").shape[0] > 0:
        eq = (1 + perf.fillna(0)).cumprod()
        eq.plot(title="Regime-Switching Strategy vs Equal-Weight Benchmark", linewidth=1.5)
        plt.tight_layout()
        plt.show()
    else:
        print("No backtest rows available (too little post-window data). You can still use the 'today' signal below.")

    # Today’s signal
    if not named.empty:
        today = named.index[-1]
        regime_today = named.iloc[-1]
        weights_today = POLICY.get(regime_today, {a: 0.0 for a in ASSETS})
        print(f"\nAs of {today.date()} (detected regime: {regime_today}):")
        for k, v in weights_today.items():
            print(f"  {k}: {v:.0%}")
    else:
        print("\nNo regime signal available yet — try widening the date range.")

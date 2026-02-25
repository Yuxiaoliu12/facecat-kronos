"""
Walk-Forward Backtester

Runs the full 4-layer pipeline with quarterly retraining (expanding window),
no lookahead bias.  Reports layer attribution and comparison benchmarks.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig
from screener.data_pipeline import (
    init_qlib,
    load_alpha158_factors,
    load_alpha158_labels,
    load_market_regime_features,
    load_raw_ohlcv,
)
from screener.factor_timing_model import FactorTimingModel
from screener.technical_ranker import TechnicalRanker
from screener.kronos_screener import KronosScreener
from screener.paper_trader import PaperTrader


class WalkForwardBacktester:
    """Walk-forward backtester with quarterly retraining windows."""

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()

        # Models
        self.layer1 = FactorTimingModel(self.cfg)
        self.layer2 = TechnicalRanker(self.cfg)
        self.layer3 = KronosScreener(self.cfg)
        self.trader = PaperTrader(self.cfg)

        # Data caches
        self._alpha158: pd.DataFrame | None = None
        self._labels: pd.Series | None = None
        self._regime: pd.DataFrame | None = None
        self._ohlcv: dict[str, pd.DataFrame] | None = None
        self._calendar: pd.DatetimeIndex | None = None

    # ── Data Loading ─────────────────────────────────────────────────────

    def load_data(self):
        """Load all required data."""
        print("=" * 60)
        print("Loading data…")
        print("=" * 60)
        init_qlib(self.cfg)

        self._alpha158 = load_alpha158_factors(self.cfg)
        self._labels = load_alpha158_labels(self.cfg)
        self._regime = load_market_regime_features(self.cfg)

        from qlib.data import D
        self._calendar = pd.DatetimeIndex(D.calendar(
            start_time=self.cfg.train_start, end_time=self.cfg.backtest_end
        ))

        # Load OHLCV for all universe stocks (needed for Layer 2 features + paper trading)
        all_symbols = list(self._alpha158.index.get_level_values("instrument").unique())
        print(f"Loading OHLCV for {len(all_symbols)} symbols…")
        self._ohlcv = load_raw_ohlcv(
            all_symbols, self.cfg.train_start, self.cfg.backtest_end, self.cfg
        )
        print(f"OHLCV loaded for {len(self._ohlcv)} symbols.")

    # ── Retraining Windows ───────────────────────────────────────────────

    def _generate_retrain_windows(self) -> list[dict]:
        """Generate quarterly expanding-window retrain schedule.

        Returns list of dicts with keys:
          train_end, val_start, val_end, test_start, test_end
        """
        windows = []
        backtest_start = pd.Timestamp(self.cfg.backtest_start)
        backtest_end = pd.Timestamp(self.cfg.backtest_end)

        # Generate quarterly boundaries within backtest period
        quarters = pd.date_range(backtest_start, backtest_end, freq="QS")
        if len(quarters) == 0:
            quarters = pd.DatetimeIndex([backtest_start])

        for i, q_start in enumerate(quarters):
            q_end = quarters[i + 1] - pd.Timedelta(days=1) if i + 1 < len(quarters) else backtest_end

            # Expanding window: train starts from cfg.train_start, ends 6 months before test
            train_end_approx = q_start - pd.DateOffset(months=6)
            val_start = train_end_approx + pd.Timedelta(days=1)
            val_end = q_start - pd.Timedelta(days=1)

            windows.append({
                "train_end": train_end_approx.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "test_start": q_start.strftime("%Y-%m-%d"),
                "test_end": q_end.strftime("%Y-%m-%d"),
            })

        return windows

    # ── Layer 1+2 Training ───────────────────────────────────────────────

    def _train_layer1(self, train_end: str):
        """Train/retrain Layer 1 factor timing model."""
        print(f"\n  Training Layer 1 (train_end={train_end})…")
        X, Y = self.layer1.build_training_data(self._alpha158, self._regime)
        self.layer1.train(X, Y, train_end=train_end)

    def _train_layer2(self, train_end: str):
        """Train/retrain Layer 2 technical ranker."""
        print(f"\n  Training Layer 2 (train_end={train_end})…")
        end_ts = pd.Timestamp(train_end)

        # Build training data: sample dates from training period
        train_dates = self._calendar[
            (self._calendar >= pd.Timestamp(self.cfg.train_start))
            & (self._calendar <= end_ts)
        ]
        # Subsample for speed (every 5th trading day)
        train_dates = train_dates[::5]

        # Forward returns for labels
        fwd_ret = self._compute_forward_returns()

        X, y, group = self.layer2.build_training_data(
            self._ohlcv, list(train_dates), fwd_ret
        )
        if len(X) > 0:
            self.layer2.train(X, y, group)
        else:
            print("  Warning: no training data for Layer 2")

    def _compute_forward_returns(self) -> pd.DataFrame:
        """Compute 5-day forward returns for all stocks (for Layer 2 labels)."""
        close_dict = {}
        for sym, df in self._ohlcv.items():
            close_dict[sym] = df["close"]
        close_df = pd.DataFrame(close_dict)
        fwd = close_df.shift(-self.cfg.layer2_forward_days) / close_df - 1
        return fwd

    # ── Daily Pipeline ───────────────────────────────────────────────────

    def _run_daily_pipeline(
        self,
        date: pd.Timestamp,
        run_kronos: bool = True,
    ) -> dict:
        """Execute the full 4-layer pipeline for one trading day.

        Returns dict with layer outputs for attribution analysis.
        """
        result = {"date": date, "layer1": [], "layer2": [], "layer3": []}

        # Layer 1: Factor Timing → top 200
        try:
            layer1_picks = self.layer1.select_top(date, alpha158_df=self._alpha158)
        except Exception as e:
            print(f"  Layer 1 failed on {date}: {e}")
            layer1_picks = []
        result["layer1"] = layer1_picks

        if not layer1_picks:
            return result

        # Layer 2: Technical Ranking → top 30
        try:
            layer2_picks = self.layer2.select_top(
                self._ohlcv, layer1_picks, date, include_news=False  # no news in backtest
            )
        except Exception as e:
            print(f"  Layer 2 failed on {date}: {e}")
            layer2_picks = layer1_picks[:self.cfg.layer2_top_n]
        result["layer2"] = layer2_picks

        # Layer 3: Kronos → top 5
        layer3_picks = layer2_picks[:self.cfg.layer3_top_n]
        kronos_preds = {}
        if run_kronos and layer2_picks:
            try:
                scores = self.layer3.screen_stocks(
                    self._ohlcv, layer2_picks, date
                )
                if not scores.empty:
                    layer3_picks = list(scores.head(self.cfg.layer3_top_n).index)
                    for sym in layer3_picks:
                        pred = self.layer3.get_prediction(sym)
                        if pred and "pred_df" in pred:
                            kronos_preds[sym] = pred["pred_df"]
            except Exception as e:
                print(f"  Layer 3 (Kronos) failed on {date}: {e}")
        result["layer3"] = layer3_picks
        result["kronos_preds"] = kronos_preds

        return result

    # ── Full Backtest ────────────────────────────────────────────────────

    def run(
        self,
        run_kronos: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Run the full walk-forward backtest.

        Returns:
            Dict with keys: metrics, nav_series, trade_log, layer_attribution.
        """
        if self._alpha158 is None:
            self.load_data()

        windows = self._generate_retrain_windows()
        print(f"\nBacktest windows: {len(windows)}")
        for w in windows:
            print(f"  Train→{w['train_end']}  Val:{w['val_start']}→{w['val_end']}  "
                  f"Test:{w['test_start']}→{w['test_end']}")

        self.trader.reset()
        layer_outputs = []

        for wi, window in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"Window {wi+1}/{len(windows)}: test {window['test_start']} → {window['test_end']}")
            print(f"{'='*60}")

            # Retrain models
            self._train_layer1(window["train_end"])
            self._train_layer2(window["train_end"])

            # Load Kronos model once per window
            if run_kronos:
                try:
                    self.layer3.load_model()
                except Exception as e:
                    print(f"  Kronos model load failed: {e}")
                    run_kronos = False

            # Test period
            test_dates = self._calendar[
                (self._calendar >= pd.Timestamp(window["test_start"]))
                & (self._calendar <= pd.Timestamp(window["test_end"]))
            ]

            for di, date in enumerate(test_dates):
                if verbose and di % 20 == 0:
                    print(f"  Day {di+1}/{len(test_dates)}: {date.date()}")

                # Run pipeline
                pipeline_out = self._run_daily_pipeline(date, run_kronos=run_kronos)
                layer_outputs.append(pipeline_out)

                # Get today's + prev day's OHLCV for paper trader
                ohlcv_today = {}
                ohlcv_prev = {}
                for sym in set(pipeline_out.get("layer3", []) +
                               ([self.trader.position.symbol] if self.trader.position else [])):
                    df = self._ohlcv.get(sym)
                    if df is None:
                        continue
                    if date in df.index:
                        ohlcv_today[sym] = df.loc[date]
                    # Previous day
                    prev_dates = df.index[df.index < date]
                    if len(prev_dates) > 0:
                        ohlcv_prev[sym] = df.loc[prev_dates[-1]]

                # Paper trader daily update
                self.trader.daily_update(
                    date=date,
                    ranked_symbols=pipeline_out.get("layer3", []),
                    ohlcv_today=ohlcv_today,
                    ohlcv_prev=ohlcv_prev,
                    kronos_predictions=pipeline_out.get("kronos_preds", {}),
                )

            # Unload Kronos after each window to save GPU memory
            if run_kronos:
                self.layer3.unload_model()

        # ── Results ──────────────────────────────────────────────────────
        metrics = self.trader.get_metrics()
        nav = self.trader.get_nav_series()

        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")

        # Layer attribution
        attribution = self._compute_layer_attribution(layer_outputs)

        return {
            "metrics": metrics,
            "nav_series": nav,
            "trade_log": self.trader.trade_log,
            "layer_attribution": attribution,
            "layer_outputs": layer_outputs,
        }

    # ── Layer Attribution ────────────────────────────────────────────────

    def _compute_layer_attribution(self, layer_outputs: list[dict]) -> dict:
        """Measure marginal alpha contribution of each layer.

        Computes the average 5-day forward return of stocks at each layer's
        cutoff to see how much each layer improves selection.
        """
        fwd_ret = self._compute_forward_returns()

        layer_returns = {"layer1": [], "layer2": [], "layer3": [], "universe": []}

        for out in layer_outputs:
            date = out["date"]
            if date not in fwd_ret.index:
                continue

            day_ret = fwd_ret.loc[date].dropna()
            if day_ret.empty:
                continue

            # Universe average
            layer_returns["universe"].append(day_ret.mean())

            for layer_name in ["layer1", "layer2", "layer3"]:
                picks = out.get(layer_name, [])
                if picks:
                    common = [s for s in picks if s in day_ret.index]
                    if common:
                        layer_returns[layer_name].append(day_ret.loc[common].mean())

        attribution = {}
        for name, rets in layer_returns.items():
            if rets:
                attribution[name] = {
                    "mean_5d_return": float(np.mean(rets)),
                    "std": float(np.std(rets)),
                    "n_days": len(rets),
                }
            else:
                attribution[name] = {"mean_5d_return": 0.0, "std": 0.0, "n_days": 0}

        print("\nLayer Attribution (avg 5-day forward return of selected stocks):")
        for name, stats in attribution.items():
            print(f"  {name:>10}: {stats['mean_5d_return']*100:.3f}% "
                  f"(±{stats['std']*100:.3f}%, n={stats['n_days']})")

        return attribution

    # ── Persistence ──────────────────────────────────────────────────────

    def save_results(self, results: dict, path: str | None = None):
        """Save backtest results to disk."""
        path = path or os.path.join(self.cfg.drive_root, "backtest_results.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved → {path}")

"""
Walk-Forward Backtester

Runs the full 4-layer pipeline with rolling-window initial training and
quarterly fine-tuning (warm-start XGBoost).  No lookahead bias.
Reports layer attribution and comparison benchmarks.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig
from screener.data_pipeline import (
    init_data,
    load_alpha158_factors,
    load_alpha158_labels,
    load_alpha158_year,
    load_market_regime_features,
    load_raw_ohlcv,
    get_calendar,
    _get_ohlcv_cache,
)
from screener.factor_timing_model import FactorTimingModel
from screener.technical_ranker import TechnicalRanker
from screener.paper_trader import PaperTrader


class WalkForwardBacktester:
    """Walk-forward backtester with quarterly retraining windows."""

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()

        # Models
        self.layer1 = FactorTimingModel(self.cfg)
        self.layer2 = TechnicalRanker(self.cfg)
        self.layer3 = None  # lazy-loaded when run_kronos=True
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
        init_data(self.cfg)

        # Alpha158 is loaded year-by-year inside Layer 1 to limit memory.
        # We set it to None here; Layer 1 handles its own loading.
        self._alpha158 = None
        self._labels = None
        self._regime = load_market_regime_features(self.cfg)

        self._calendar = get_calendar(
            self.cfg, self.cfg.train_start, self.cfg.backtest_end
        )

        # Load OHLCV for all universe stocks (needed for Layer 2 features + paper trading)
        all_symbols = list(_get_ohlcv_cache(self.cfg).keys())
        print(f"Loading OHLCV for {len(all_symbols)} symbols…")
        self._ohlcv = load_raw_ohlcv(
            all_symbols, self.cfg.train_start, self.cfg.backtest_end, self.cfg
        )
        print(f"OHLCV loaded for {len(self._ohlcv)} symbols.")

        # Precompute Layer 2 technical features for all stocks (cache for fast lookup)
        self.layer2.precompute_features(self._ohlcv)

    # ── Retraining Windows ───────────────────────────────────────────────

    def _generate_retrain_windows(self) -> list[dict]:
        """Generate quarterly rolling-window schedule.

        Window 1: initial full training on [backtest_start - train_years, backtest_start - 1d]
        Windows 2+: fine-tune on previous quarter's data only.

        Returns list of dicts with keys:
          train_start, train_end, test_start, test_end, mode
        """
        cfg = self.cfg
        windows = []
        backtest_start = pd.Timestamp(cfg.backtest_start)
        backtest_end = pd.Timestamp(cfg.backtest_end)

        quarters = pd.date_range(backtest_start, backtest_end, freq="QS")
        if len(quarters) == 0:
            quarters = pd.DatetimeIndex([backtest_start])

        for i, q_start in enumerate(quarters):
            q_end = (
                quarters[i + 1] - pd.Timedelta(days=1)
                if i + 1 < len(quarters)
                else backtest_end
            )

            if i == 0:
                # Initial: full training on [backtest_start - train_years, backtest_start - 1d]
                train_start = q_start - pd.DateOffset(years=cfg.train_years)
                train_end = q_start - pd.Timedelta(days=1)
                mode = "initial"
            else:
                # Fine-tune: train on previous quarter's data only
                train_start = quarters[i - 1]
                train_end = q_start - pd.Timedelta(days=1)
                mode = "finetune"

            windows.append({
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": q_start.strftime("%Y-%m-%d"),
                "test_end": q_end.strftime("%Y-%m-%d"),
                "mode": mode,
            })

        return windows

    # ── Layer 1+2 Training ───────────────────────────────────────────────

    def _train_layer1(self, train_start: str, train_end: str):
        """Full initial training for Layer 1."""
        print(f"\n  Training Layer 1 ({train_start} → {train_end})…")
        X, Y = self.layer1.build_training_data(
            self._alpha158, self._regime,
            train_start=train_start, train_end=train_end,
        )
        self.layer1.train(X, Y, train_start=train_start, train_end=train_end)

    def _finetune_layer1(self, train_start: str, train_end: str):
        """Fine-tune Layer 1 on new quarter's data."""
        print(f"\n  Fine-tuning Layer 1 ({train_start} → {train_end})…")
        X, Y = self.layer1.build_training_data(
            self._alpha158, self._regime,
            train_start=train_start, train_end=train_end,
        )
        self.layer1.finetune(X, Y)

    def _get_layer2_dates(self, train_start: str, train_end: str) -> list:
        """Get training dates for Layer 2 with leakage trimming and subsampling."""
        start_ts = pd.Timestamp(train_start)
        end_ts = pd.Timestamp(train_end)
        train_dates = self._calendar[
            (self._calendar >= start_ts) & (self._calendar <= end_ts)
        ]
        # Leakage trim: drop last N dates whose forward returns peek into test
        trim = self.cfg.forward_horizon_days
        if len(train_dates) > trim:
            train_dates = train_dates[:-trim]
        # Subsample for speed (every 5th trading day)
        train_dates = train_dates[::5]
        return list(train_dates)

    def _train_layer2(self, train_start: str, train_end: str):
        """Full initial training for Layer 2."""
        print(f"\n  Training Layer 2 ({train_start} → {train_end})…")
        train_dates = self._get_layer2_dates(train_start, train_end)
        upside, downside = self._compute_forward_hl_returns()
        X, y_up, y_down = self.layer2.build_training_data(
            self._ohlcv, train_dates, upside, downside
        )
        if len(X) > 0:
            self.layer2.train(X, y_up, y_down)
        else:
            print("  Warning: no training data for Layer 2")

    def _finetune_layer2(self, train_start: str, train_end: str):
        """Fine-tune Layer 2 on new quarter's data."""
        print(f"\n  Fine-tuning Layer 2 ({train_start} → {train_end})…")
        train_dates = self._get_layer2_dates(train_start, train_end)
        upside, downside = self._compute_forward_hl_returns()
        X, y_up, y_down = self.layer2.build_training_data(
            self._ohlcv, train_dates, upside, downside
        )
        if len(X) > 0:
            self.layer2.finetune(X, y_up, y_down)
        else:
            print("  Warning: no fine-tuning data for Layer 2")

    def _compute_forward_hl_returns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute forward max-high upside and min-low downside."""
        fwd = self.cfg.layer2_forward_days  # 5
        upside_dict, downside_dict = {}, {}
        for sym, df in self._ohlcv.items():
            close, high, low = df["close"], df["high"], df["low"]
            # Forward max high over [t+1, t+fwd]: reverse → rolling max → reverse → shift
            fwd_max = high[::-1].rolling(fwd, min_periods=fwd).max()[::-1].shift(-1)
            fwd_min = low[::-1].rolling(fwd, min_periods=fwd).min()[::-1].shift(-1)
            upside_dict[sym] = fwd_max / close - 1
            downside_dict[sym] = fwd_min / close - 1
        return pd.DataFrame(upside_dict), pd.DataFrame(downside_dict)

    def _compute_forward_close_returns(self) -> pd.DataFrame:
        """Compute 5-day forward close-to-close returns (for attribution only)."""
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
        if run_kronos and layer2_picks and self.layer3 is not None:
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
        if self._ohlcv is None:
            self.load_data()

        windows = self._generate_retrain_windows()
        print(f"\nBacktest windows: {len(windows)}")
        for w in windows:
            print(f"  [{w['mode']:>8}] Train:{w['train_start']}→{w['train_end']}  "
                  f"Test:{w['test_start']}→{w['test_end']}")

        self.trader.reset()
        layer_outputs = []

        for wi, window in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"Window {wi+1}/{len(windows)} [{window['mode']}]: "
                  f"test {window['test_start']} → {window['test_end']}")
            print(f"{'='*60}")

            # Train or fine-tune models
            if window["mode"] == "initial":
                self._train_layer1(window["train_start"], window["train_end"])
                self._train_layer2(window["train_start"], window["train_end"])
            else:
                self._finetune_layer1(window["train_start"], window["train_end"])
                self._finetune_layer2(window["train_start"], window["train_end"])

            # Load Kronos model once per window (lazy import)
            if run_kronos:
                try:
                    if self.layer3 is None:
                        from screener.kronos_screener import KronosScreener
                        self.layer3 = KronosScreener(self.cfg)
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
                )

            # Unload Kronos after each window to save GPU memory
            if run_kronos and self.layer3 is not None:
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
        fwd_ret = self._compute_forward_close_returns()

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

"""
Screener configuration — all hyperparameters in one place.

Usage:
    from screener.config import ScreenerConfig
    cfg = ScreenerConfig()
"""

import os
from dataclasses import dataclass, field


@dataclass
class ScreenerConfig:
    # ── Data Sources ─────────────────────────────────────────────────────
    ohlcv_pickle_path: str = "data/ohlcv_all_a.pkl"
    benchmark_pickle_path: str = "data/benchmark_000905.pkl"
    benchmark: str = "sh.000905"

    # ── Layer 1 — Factor Timing (XGBoost) ────────────────────────────────
    layer1_top_n: int = 200
    layer1_xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 42,
    })
    layer1_forward_days: int = 5  # IC label horizon

    # ── Layer 2 — Technical Ranker (single XGBRegressor) ─────────────────
    layer2_top_n: int = 30
    layer2_xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 42,
    })
    layer2_forward_days: int = 5  # ranking label horizon

    # ── Layer 3 — Kronos ─────────────────────────────────────────────────
    layer3_top_n: int = 5
    kronos_pred_len: int = 10
    kronos_sample_count: int = 5
    kronos_T: float = 0.6
    kronos_top_p: float = 0.9
    kronos_top_k: int = 0
    kronos_lookback: int = 90
    kronos_max_context: int = 512
    kronos_clip: float = 5.0

    # Paths to Kronos model weights (update to your fine-tuned checkpoints)
    kronos_tokenizer_path: str = ""
    kronos_predictor_path: str = ""

    # ── Layer 4 — Paper Trading ──────────────────────────────────────────
    initial_capital: float = 1_000_000.0
    buy_commission: float = 0.00025       # 0.025%
    sell_commission: float = 0.00025      # 0.025%
    stamp_tax: float = 0.0005            # 0.05%  (sell only)
    max_hold_days: int = 5
    lot_size: int = 100                  # A-share minimum lot
    tp_pct: float = 0.05                 # take-profit threshold (5%)
    sl_pct: float = 0.05                 # stop-loss threshold (5%)

    # Limit-up/down thresholds by board
    limit_main: float = 0.10    # 主板
    limit_gem_star: float = 0.20  # 创业板 / 科创板
    limit_ipo_gem_star: float = 0.30  # first 5 days IPO on 创业板/科创板

    # ── Walk-Forward Settings ─────────────────────────────────────────────
    backtest_start: str = "2017-01-01"
    backtest_end: str = "2017-12-31"
    train_years: int = 2                    # initial rolling window size
    forward_horizon_days: int = 5           # leakage trim (drop last N train rows)
    retrain_freq: str = "Q"                 # quarterly

    # Fine-tuning (warm-start) hyperparameters
    finetune_n_estimators: int = 50         # trees added per fine-tune round
    finetune_learning_rate: float = 0.02    # conservative LR for updates

    # Exponential sample weighting: recent data weighted higher
    sample_weight_halflife_days: int = 120  # half-weight every ~6 months; 0 = disabled

    # Auto-computed from backtest_start - train_years (set in __post_init__)
    train_start: str = ""
    train_end: str = ""

    # ── Persistence ──────────────────────────────────────────────────────
    drive_root: str = "output/screener"
    alpha158_cache: str = ""    # auto-set in __post_init__
    model_cache: str = ""       # auto-set in __post_init__

    # ── News Scorer ──────────────────────────────────────────────────────
    news_model_name: str = "yiyanghkust/finbert-tone"
    news_max_headlines: int = 20  # per stock per day
    news_batch_size: int = 64

    # Policy keywords for flag detection
    policy_keywords: list = field(default_factory=lambda: [
        "央行", "监管", "政策", "降息", "加息", "降准",
        "财政", "证监会", "银保监", "国务院", "发改委",
    ])

    def __post_init__(self):
        import pandas as pd

        self.alpha158_cache = os.path.join(self.drive_root, "alpha158_cache.pkl")
        self.model_cache = os.path.join(self.drive_root, "models")
        if not self.kronos_tokenizer_path:
            self.kronos_tokenizer_path = os.path.join(self.drive_root, "models", "kronos_tokenizer")
        if not self.kronos_predictor_path:
            self.kronos_predictor_path = os.path.join(self.drive_root, "models", "kronos_predictor")

        # Auto-compute train bounds from backtest_start - train_years
        bt_start = pd.Timestamp(self.backtest_start)
        if not self.train_start:
            self.train_start = (bt_start - pd.DateOffset(years=self.train_years)).strftime("%Y-%m-%d")
        if not self.train_end:
            self.train_end = (bt_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

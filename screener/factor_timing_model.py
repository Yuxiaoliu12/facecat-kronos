"""
Layer 1: Factor Timing Model  (CSI1000 → ~200 stocks)

Learns which Alpha158 factor *categories* are predictive of returns given
the current calendar / market regime.  Outputs a per-stock composite score
by weighting factor categories according to predicted importance.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from screener.config import ScreenerConfig
from screener.utils import FACTOR_CATEGORIES, robust_zscore, group_features_by_category
from screener.data_pipeline import (
    load_alpha158_factors,
    load_alpha158_labels,
    load_market_regime_features,
    compute_lagged_factor_ic,
)


class FactorTimingModel:
    """XGBoost multi-output regressor that predicts per-category forward IC."""

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()
        self.model: MultiOutputRegressor | None = None
        self.feature_names: list[str] = []
        self.target_names: list[str] = [f"fwd_ic_{c}" for c in FACTOR_CATEGORIES]
        self._alpha158_df: pd.DataFrame | None = None  # cached

    # ── Training ─────────────────────────────────────────────────────────

    def build_training_data(
        self,
        alpha158_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build (X, Y) where each row is one trading day.

        X: regime features + lagged factor IC  (~60 dims)
        Y: forward IC per factor category      (~10 dims)
        """
        cfg = self.cfg
        if alpha158_df is None:
            alpha158_df = load_alpha158_factors(cfg, cfg.train_start, cfg.backtest_end)
        self._alpha158_df = alpha158_df

        if regime_df is None:
            regime_df = load_market_regime_features(cfg, cfg.train_start, cfg.backtest_end)

        # Compute forward returns (5-day) for IC labels
        labels = load_alpha158_labels(cfg, cfg.train_start, cfg.backtest_end)

        # Lagged IC (trailing 20-day rolling IC per category)
        lagged_ic = compute_lagged_factor_ic(alpha158_df, labels, lookback=20)

        # Forward IC: per-day Spearman(category_mean_score, actual_forward_return)
        fwd_ic = self._compute_forward_ic(alpha158_df, labels)

        # Merge X
        X = regime_df.join(lagged_ic, how="inner")
        Y = fwd_ic.reindex(X.index)
        mask = Y.notna().all(axis=1) & X.notna().all(axis=1)
        X, Y = X.loc[mask], Y.loc[mask]
        self.feature_names = list(X.columns)
        return X, Y

    def _compute_forward_ic(
        self, alpha158_df: pd.DataFrame, returns: pd.Series
    ) -> pd.DataFrame:
        """Compute daily forward IC per factor category (the label for training)."""
        feature_groups = group_features_by_category(list(alpha158_df.columns))
        dates = sorted(alpha158_df.index.get_level_values("datetime").unique())

        records = []
        for dt in dates:
            row = {"datetime": dt}
            try:
                day_f = alpha158_df.xs(dt, level="datetime")
                day_r = returns.xs(dt, level="datetime") if isinstance(returns.index, pd.MultiIndex) else returns.loc[dt]
            except KeyError:
                continue
            for cat in FACTOR_CATEGORIES:
                feats = feature_groups.get(cat, [])
                if not feats:
                    row[f"fwd_ic_{cat}"] = 0.0
                    continue
                cat_score = day_f[feats].mean(axis=1)
                valid = pd.DataFrame({"s": cat_score, "r": day_r}).dropna()
                if len(valid) < 10:
                    row[f"fwd_ic_{cat}"] = 0.0
                else:
                    row[f"fwd_ic_{cat}"] = valid["s"].corr(valid["r"], method="spearman")
            records.append(row)

        return pd.DataFrame(records).set_index("datetime")

    def train(
        self,
        X: pd.DataFrame | None = None,
        Y: pd.DataFrame | None = None,
        train_end: str | None = None,
    ):
        """Fit the multi-output XGBoost model on training period."""
        if X is None or Y is None:
            X, Y = self.build_training_data()

        train_end = pd.Timestamp(train_end or self.cfg.train_end)
        mask = X.index <= train_end
        X_train, Y_train = X.loc[mask], Y.loc[mask]
        print(f"Layer 1 training: {len(X_train)} days, {len(self.feature_names)} features, "
              f"{len(self.target_names)} targets")

        base = XGBRegressor(**self.cfg.layer1_xgb_params)
        self.model = MultiOutputRegressor(base)
        self.model.fit(X_train.values, Y_train.values)
        print("Layer 1 model trained.")

    def validate(self, X: pd.DataFrame, Y: pd.DataFrame, val_start: str, val_end: str) -> dict:
        """Evaluate predicted IC vs actual forward IC on validation set."""
        mask = (X.index >= pd.Timestamp(val_start)) & (X.index <= pd.Timestamp(val_end))
        X_val, Y_val = X.loc[mask], Y.loc[mask]
        Y_pred = self.model.predict(X_val.values)
        Y_pred_df = pd.DataFrame(Y_pred, index=Y_val.index, columns=self.target_names)

        corrs = {}
        for col in self.target_names:
            valid = pd.DataFrame({"pred": Y_pred_df[col], "actual": Y_val[col]}).dropna()
            if len(valid) >= 10:
                corrs[col] = spearmanr(valid["pred"], valid["actual"]).correlation
            else:
                corrs[col] = float("nan")
        print(f"Layer 1 validation IC (predicted vs actual forward IC):")
        for k, v in corrs.items():
            print(f"  {k}: {v:.4f}")
        return corrs

    # ── Inference ────────────────────────────────────────────────────────

    def score_stocks(
        self,
        date: pd.Timestamp,
        alpha158_df: pd.DataFrame | None = None,
        regime_row: pd.Series | None = None,
    ) -> pd.Series:
        """Score all stocks on a given date, return top-N symbols.

        1. Predict factor category weights from regime features.
        2. Compute weighted composite score per stock.
        3. Return per-stock score (higher = better).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        if alpha158_df is None:
            alpha158_df = self._alpha158_df
        if alpha158_df is None:
            alpha158_df = load_alpha158_factors(self.cfg)
            self._alpha158_df = alpha158_df

        # Get regime features for this date
        if regime_row is None:
            regime_df = load_market_regime_features(self.cfg, str(date.date()), str(date.date()))
            regime_row = regime_df.iloc[-1] if len(regime_df) > 0 else pd.Series(dtype=float)

        # Align features
        x = regime_row.reindex(self.feature_names, fill_value=0).values.reshape(1, -1)
        predicted_weights = self.model.predict(x)[0]  # shape (n_categories,)

        # Normalise weights to [0, 1] via softmax-like transform
        weights = np.exp(predicted_weights)
        weights = weights / (weights.sum() + 1e-9)
        weight_map = dict(zip(FACTOR_CATEGORIES, weights))

        # Get per-stock Alpha158 for this date
        try:
            day_factors = alpha158_df.xs(date, level="datetime")
        except KeyError:
            return pd.Series(dtype=float)

        feature_groups = group_features_by_category(list(day_factors.columns))
        composite = pd.Series(0.0, index=day_factors.index)
        for cat, w in weight_map.items():
            feats = feature_groups.get(cat, [])
            if feats:
                composite += w * day_factors[feats].mean(axis=1)

        return composite.sort_values(ascending=False)

    def select_top(self, date: pd.Timestamp, **kwargs) -> list[str]:
        """Return top-N stock symbols for a given date."""
        scores = self.score_stocks(date, **kwargs)
        return list(scores.head(self.cfg.layer1_top_n).index)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer1_factor_timing.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        print(f"Layer 1 model saved → {path}")

    def load(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer1_factor_timing.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        print(f"Layer 1 model loaded ← {path}")

"""
Layer 2: Technical Ranking Model  (200 → ~30 stocks)

Uses XGBRanker with ``rank:pairwise`` (LambdaMART) to learn cross-sectional
stock rankings from weekly technical indicators + news sentiment.
"""

import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from screener.config import ScreenerConfig
from screener.news_scorer import NewsScorer


class TechnicalRanker:
    """XGBRanker that ranks stocks by predicted 5-day forward return."""

    TECH_FEATURES = [
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "rsi_5",
        "ma5_slope", "ma20_slope", "ma60_slope",
        "bb_position",
        "volume_trend",
        "mom_5", "mom_10", "mom_20",
        "atr_14",
        "obv_slope",
    ]
    NEWS_FEATURES = ["news_sentiment", "news_significance", "policy_flag"]
    ALL_FEATURES = TECH_FEATURES + NEWS_FEATURES

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()
        self.model: XGBRanker | None = None
        self.news_scorer: NewsScorer | None = None

    # ── Feature Computation ──────────────────────────────────────────────

    @staticmethod
    def compute_technical_features(df: pd.DataFrame) -> pd.Series:
        """Compute technical features from an OHLCV DataFrame for one stock.

        Expects columns: open, high, low, close, volume (or vol).
        Returns a Series with feature values for the *last* row.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else df["vol"]

        feats = {}

        # MACD(12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        feats["macd"] = macd_line.iloc[-1]
        feats["macd_signal"] = signal_line.iloc[-1]
        feats["macd_hist"] = (macd_line - signal_line).iloc[-1]

        # RSI(14), RSI(5)
        for period in [14, 5]:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / (loss + 1e-9)
            feats[f"rsi_{period}"] = (100 - 100 / (1 + rs)).iloc[-1]

        # MA slopes (normalised by price)
        for w in [5, 20, 60]:
            ma = close.rolling(w).mean()
            slope = (ma.iloc[-1] - ma.iloc[-min(w, 5)]) / (close.iloc[-1] + 1e-9)
            feats[f"ma{w}_slope"] = slope

        # Bollinger Band position: (close - lower) / (upper - lower)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bb_range = upper.iloc[-1] - lower.iloc[-1]
        feats["bb_position"] = (close.iloc[-1] - lower.iloc[-1]) / (bb_range + 1e-9)

        # Volume trend (5-day MA / 20-day MA)
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        feats["volume_trend"] = (vol_ma5.iloc[-1] / (vol_ma20.iloc[-1] + 1e-9))

        # Momentum (return over N days)
        for d in [5, 10, 20]:
            feats[f"mom_{d}"] = (close.iloc[-1] / close.iloc[-d] - 1) if len(close) > d else 0.0

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        feats["atr_14"] = tr.rolling(14).mean().iloc[-1] / (close.iloc[-1] + 1e-9)

        # OBV slope (normalised)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / (abs(obv.iloc[-5]) + 1e-9) if len(obv) > 5 else 0.0
        feats["obv_slope"] = obv_slope

        return pd.Series(feats)

    def compute_features_for_stocks(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        date: pd.Timestamp | None = None,
        symbols: list[str] | None = None,
        include_news: bool = True,
    ) -> pd.DataFrame:
        """Compute technical + news features for a list of stocks.

        Args:
            ohlcv_dict: symbol → OHLCV DataFrame.
            date: Date for which to compute features (uses latest data up to this date).
            symbols: Subset of symbols to process.  Defaults to all in ohlcv_dict.
            include_news: Whether to include news sentiment features.

        Returns:
            DataFrame indexed by symbol, one row per stock, columns = ALL_FEATURES.
        """
        symbols = symbols or list(ohlcv_dict.keys())
        records = []
        for sym in symbols:
            df = ohlcv_dict.get(sym)
            if df is None or len(df) < 60:
                continue
            if date is not None:
                df = df.loc[:date]
                if len(df) < 60:
                    continue
            try:
                feats = self.compute_technical_features(df)
                feats.name = sym
                records.append(feats)
            except Exception:
                continue

        if not records:
            return pd.DataFrame(columns=self.ALL_FEATURES)

        tech_df = pd.DataFrame(records)

        # Add news features
        if include_news:
            news_df = self._get_news_features(list(tech_df.index))
            tech_df = tech_df.join(news_df, how="left")

        # Fill missing with 0
        for col in self.ALL_FEATURES:
            if col not in tech_df.columns:
                tech_df[col] = 0.0
        tech_df = tech_df[self.ALL_FEATURES].fillna(0)
        return tech_df

    def _get_news_features(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch news sentiment features for symbols."""
        if self.news_scorer is None:
            self.news_scorer = NewsScorer(self.cfg)
        try:
            return self.news_scorer.score_batch(symbols)
        except Exception:
            # Graceful degradation — return zeros if news fetching fails
            return pd.DataFrame(0.0, index=symbols, columns=self.NEWS_FEATURES)

    # ── Training ─────────────────────────────────────────────────────────

    def build_training_data(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        dates: list[pd.Timestamp],
        forward_returns: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y, group) arrays for XGBRanker training.

        Args:
            ohlcv_dict: symbol → full OHLCV history.
            dates: List of training dates.
            forward_returns: DataFrame (datetime×symbol) of 5-day forward returns.

        Returns:
            X: (total_stocks_across_days, n_features)
            y: (total_stocks_across_days,) — cross-sectional rank label
            group: list of group sizes (one per date)
        """
        X_list, y_list, groups = [], [], []

        for dt in dates:
            feat_df = self.compute_features_for_stocks(
                ohlcv_dict, date=dt, include_news=False  # news only at inference
            )
            if feat_df.empty:
                continue

            # Get forward returns for this date
            if dt not in forward_returns.index:
                continue
            fwd = forward_returns.loc[dt]
            common = feat_df.index.intersection(fwd.dropna().index)
            if len(common) < 10:
                continue

            feat_df = feat_df.loc[common]
            fwd_common = fwd.loc[common]

            # Label = cross-sectional rank (higher return → higher rank)
            ranks = fwd_common.rank(ascending=True).values

            X_list.append(feat_df.values)
            y_list.append(ranks)
            groups.append(len(common))

        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list).astype(np.float32)
        return X, y, np.array(groups)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
    ):
        """Fit XGBRanker on pre-built training data."""
        params = dict(self.cfg.layer2_xgb_params)
        self.model = XGBRanker(**params)
        self.model.fit(X, y, group=group)
        print(f"Layer 2 model trained: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(group)} groups (days)")

    def validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        group_val: np.ndarray,
    ) -> dict:
        """Evaluate ranking quality (NDCG-like) on validation set."""
        from scipy.stats import spearmanr

        preds = self.model.predict(X_val)

        # Per-group Spearman correlation (proxy for ranking quality)
        corrs = []
        offset = 0
        for g in group_val:
            if g < 5:
                offset += g
                continue
            p = preds[offset:offset + g]
            a = y_val[offset:offset + g]
            c, _ = spearmanr(p, a)
            corrs.append(c)
            offset += g

        mean_corr = float(np.mean(corrs)) if corrs else 0.0
        print(f"Layer 2 validation: mean Spearman rank corr = {mean_corr:.4f} "
              f"(over {len(corrs)} groups)")
        return {"mean_rank_corr": mean_corr, "n_groups": len(corrs)}

    # ── Inference ────────────────────────────────────────────────────────

    def rank_stocks(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        symbols: list[str],
        date: pd.Timestamp,
        include_news: bool = True,
    ) -> pd.Series:
        """Rank a set of symbols by predicted forward return.

        Returns:
            Series indexed by symbol, sorted descending (best first).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        feat_df = self.compute_features_for_stocks(
            ohlcv_dict, date=date, symbols=symbols, include_news=include_news
        )
        if feat_df.empty:
            return pd.Series(dtype=float)

        scores = self.model.predict(feat_df.values.astype(np.float32))
        result = pd.Series(scores, index=feat_df.index, name="rank_score")
        return result.sort_values(ascending=False)

    def select_top(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        symbols: list[str],
        date: pd.Timestamp,
        **kwargs,
    ) -> list[str]:
        """Return top-N symbols."""
        scores = self.rank_stocks(ohlcv_dict, symbols, date, **kwargs)
        return list(scores.head(self.cfg.layer2_top_n).index)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer2_technical_ranker.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model}, f)
        print(f"Layer 2 model saved → {path}")

    def load(self, path: str | None = None):
        path = path or os.path.join(self.cfg.model_cache, "layer2_technical_ranker.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        print(f"Layer 2 model loaded ← {path}")

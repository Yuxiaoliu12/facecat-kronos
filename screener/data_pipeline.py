"""
Data pipeline: Qlib init, Alpha158 factor computation, regime features, raw OHLCV.

Follows the same Qlib patterns as ``finetune/qlib_data_preprocess.py``.
Alpha158 features are computed directly via D.features() with Qlib expressions,
bypassing the Alpha158 handler which has compatibility issues across Qlib versions.
"""

import os
import pickle

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader

from screener.config import ScreenerConfig
from screener.utils import (
    calendar_features_series,
    group_features_by_category,
    FACTOR_CATEGORIES,
    robust_zscore,
)


# ── Qlib Initialisation ─────────────────────────────────────────────────────

def init_qlib(cfg: ScreenerConfig | None = None):
    """Initialise Qlib (idempotent — safe to call multiple times)."""
    cfg = cfg or ScreenerConfig()
    provider = os.path.expanduser(cfg.qlib_data_path)
    qlib.init(provider_uri=provider, region=REG_CN)


# ── Alpha158-equivalent features via D.features() ──────────────────────────

def _alpha158_exprs() -> tuple[list[str], list[str]]:
    """Return (expressions, names) for Alpha158-equivalent features.

    Produces ~109 features covering all Alpha158 categories:
    momentum, trend, volatility, price_extreme, mean_reversion,
    correlation, volume_price, volume, cross_section.
    """
    exprs = []
    names = []

    # KBAR features (9)
    kbar = [
        ("($close-$open)/$open", "KMID$0"),
        ("($high-$low)/$open", "KLEN$0"),
        ("($close-$open)/($high-$low+1e-12)", "KMID2$0"),
        ("($high-Greater($open,$close))/$open", "KUP$0"),
        ("($high-Greater($open,$close))/($high-$low+1e-12)", "KUP2$0"),
        ("(Less($open,$close)-$low)/$open", "KLOW$0"),
        ("(Less($open,$close)-$low)/($high-$low+1e-12)", "KLOW2$0"),
        ("(2*$close-$high-$low)/$open", "KSFT$0"),
        ("(2*$close-$high-$low)/($high-$low+1e-12)", "KSFT2$0"),
    ]
    for expr, name in kbar:
        exprs.append(expr)
        names.append(name)

    # Rolling features for windows [5, 10, 20, 30, 60]
    for N in [5, 10, 20, 30, 60]:
        rolling = [
            # momentum / trend
            (f"Ref($close,{N})/$close", f"ROC${N}"),
            (f"Mean($close,{N})/$close", f"MA${N}"),
            # volatility
            (f"Std($close,{N})/$close", f"STD${N}"),
            # price_extreme
            (f"Max($high,{N})/$close", f"MAX${N}"),
            (f"Min($low,{N})/$close", f"MIN${N}"),
            # mean_reversion
            (f"($close-Min($low,{N}))/(Max($high,{N})-Min($low,{N})+1e-12)", f"RSV${N}"),
            # correlation
            (f"Corr($close,Log($volume+1),{N})", f"CORR${N}"),
            (f"Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),{N})", f"CORD${N}"),
            # volume_price
            (f"Mean($close>Ref($close,1),{N})", f"CNTP${N}"),
            (f"Mean($close<Ref($close,1),{N})", f"CNTN${N}"),
            (f"Mean($close>Ref($close,1),{N})-Mean($close<Ref($close,1),{N})", f"CNTD${N}"),
            (f"Sum(Greater($close-Ref($close,1),0),{N})/(Sum(Abs($close-Ref($close,1)),{N})+1e-12)", f"SUMP${N}"),
            (f"Sum(Greater(Ref($close,1)-$close,0),{N})/(Sum(Abs($close-Ref($close,1)),{N})+1e-12)", f"SUMN${N}"),
            (
                f"(Sum(Greater($close-Ref($close,1),0),{N})-Sum(Greater(Ref($close,1)-$close,0),{N}))"
                f"/(Sum(Abs($close-Ref($close,1)),{N})+1e-12)",
                f"SUMD${N}",
            ),
            # volume
            (f"Mean($volume,{N})/($volume+1e-12)", f"VMA${N}"),
            (f"Std($volume,{N})/($volume+1e-12)", f"VSTD${N}"),
            (
                f"Std(Abs($close/Ref($close,1)-1)*$volume,{N})"
                f"/(Mean(Abs($close/Ref($close,1)-1)*$volume,{N})+1e-12)",
                f"WVMA${N}",
            ),
            (f"Sum(If($close>Ref($close,1),$volume,0),{N})/(Sum($volume,{N})+1e-12)", f"VSUMP${N}"),
            (f"Sum(If($close<Ref($close,1),$volume,0),{N})/(Sum($volume,{N})+1e-12)", f"VSUMN${N}"),
            (
                f"(Sum(If($close>Ref($close,1),$volume,0),{N})-Sum(If($close<Ref($close,1),$volume,0),{N}))"
                f"/(Sum($volume,{N})+1e-12)",
                f"VSUMD${N}",
            ),
        ]
        for expr, name in rolling:
            exprs.append(expr)
            names.append(name)

    return exprs, names


def load_alpha158_factors(
    cfg: ScreenerConfig,
    start: str | None = None,
    end: str | None = None,
    *,
    cache: bool = True,
) -> pd.DataFrame:
    """Load Alpha158-equivalent cross-sectional factors for the whole universe.

    Returns a DataFrame with MultiIndex (datetime, instrument) and ~109 factor
    columns.  Results are cached to ``cfg.alpha158_cache`` on first call.
    """
    start = start or cfg.train_start
    end = end or cfg.backtest_end

    # Try cache first
    if cache and os.path.exists(cfg.alpha158_cache):
        print(f"Loading Alpha158 from cache: {cfg.alpha158_cache}")
        df = pd.read_pickle(cfg.alpha158_cache)
        mask = (df.index.get_level_values("datetime") >= pd.Timestamp(start)) & (
            df.index.get_level_values("datetime") <= pd.Timestamp(end)
        )
        return df.loc[mask]

    print("Computing Alpha158 factors via D.features() (this may take a few minutes)…")
    exprs, feat_names = _alpha158_exprs()

    df = D.features(cfg.universe, exprs, start_time=start, end_time=end, freq="day")
    df.columns = feat_names

    # Cross-sectional RobustZScore normalisation per day
    print("Applying cross-sectional RobustZScore normalisation…")

    def _cs_robust_zscore(group, clip=3.0):
        median = group.median()
        mad = (group - median).abs().median()
        return ((group - median) / (1.4826 * mad + 1e-9)).clip(-clip, clip)

    df = df.groupby(level="datetime").transform(_cs_robust_zscore)
    df = df.fillna(0)

    # Persist
    if cache:
        os.makedirs(os.path.dirname(cfg.alpha158_cache), exist_ok=True)
        df.to_pickle(cfg.alpha158_cache)
        print(f"Alpha158 cached → {cfg.alpha158_cache}")

    return df


def load_alpha158_labels(
    cfg: ScreenerConfig,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load forward return labels (CSRankNorm'd)."""
    start = start or cfg.train_start
    end = end or cfg.backtest_end

    # Load close prices
    close_df = D.features(cfg.universe, ["$close"], start_time=start, end_time=end, freq="day")
    close_df.columns = ["close"]

    # Forward N-day return per stock
    fwd = cfg.layer1_forward_days
    fwd_ret = close_df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-fwd) / x - 1
    )
    fwd_ret = fwd_ret.dropna()

    # Cross-sectional rank normalisation per day (centre around 0)
    labels = fwd_ret.groupby(level="datetime").transform(
        lambda g: g.rank(pct=True) - 0.5
    )
    return labels


# ── Market Regime Features ───────────────────────────────────────────────────

def load_market_regime_features(
    cfg: ScreenerConfig,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Compute market-level regime features per trading day.

    Features (~40-60 dims):
      - Calendar (5): month, quarter, day_of_week, day_of_month, days_to_quarter_end
      - Market (3): 20d index return, 20d volatility, market breadth
      - Sector (≤28): 20d return per ShenWan L1 sector
      - Lagged factor IC (≤10): rolling IC per factor category over past 20d
    """
    start = start or cfg.train_start
    end = end or cfg.backtest_end

    # Calendar → trading day calendar from Qlib
    cal = D.calendar(start_time=start, end_time=end)
    cal_idx = pd.DatetimeIndex(cal)
    cal_df = calendar_features_series(cal_idx)

    # Market-level features from benchmark index
    benchmark = cfg.benchmark
    index_fields = ["$close", "$open", "$high", "$low", "$volume"]
    idx_df = D.features(
        [benchmark], index_fields, start_time=start, end_time=end, freq="day"
    )
    # Flatten MultiIndex
    if isinstance(idx_df.index, pd.MultiIndex):
        idx_df = idx_df.droplevel("instrument")
    idx_df.columns = ["idx_close", "idx_open", "idx_high", "idx_low", "idx_volume"]

    market = pd.DataFrame(index=cal_idx)
    market["idx_ret_20d"] = idx_df["idx_close"].pct_change(20)
    market["idx_vol_20d"] = idx_df["idx_close"].pct_change().rolling(20).std()

    # Market breadth: % of stocks above their 20-day MA
    # (computed from universe close prices)
    breadth = _compute_market_breadth(cfg, start, end, cal_idx)
    market["market_breadth"] = breadth

    # Sector returns (ShenWan L1 — we proxy via the $close of sector ETFs or
    # compute sector-mean returns from stock data).
    sector_df = _compute_sector_returns(cfg, start, end, cal_idx)

    # Merge everything
    regime = cal_df.join(market, how="left").join(sector_df, how="left")
    regime = regime.fillna(method="ffill").fillna(0)
    return regime


def _compute_market_breadth(
    cfg: ScreenerConfig, start: str, end: str, cal_idx: pd.DatetimeIndex
) -> pd.Series:
    """Fraction of universe stocks whose close > MA20."""
    close_fields = ["$close"]
    close_df = D.features(
        cfg.universe, close_fields, start_time=start, end_time=end, freq="day"
    )
    close_df.columns = ["close"]

    # Compute per-stock MA20 then compare
    close_unstacked = close_df["close"].unstack("instrument")
    ma20 = close_unstacked.rolling(20).mean()
    above = (close_unstacked > ma20).mean(axis=1)  # fraction above
    above = above.reindex(cal_idx)
    return above


def _compute_sector_returns(
    cfg: ScreenerConfig, start: str, end: str, cal_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Compute 20-day returns for ShenWan L1 sectors.

    Since Qlib doesn't natively provide sector classification, we use a simple
    proxy: compute the equal-weight 20-day return for each exchange-board
    grouping (by code prefix).
    """
    close_df = D.features(
        cfg.universe, ["$close"], start_time=start, end_time=end, freq="day"
    )
    close_df.columns = ["close"]
    close_unstacked = close_df["close"].unstack("instrument")

    # Group stocks by code prefix (first 3 digits) as a lightweight sector proxy
    sector_groups: dict[str, list[str]] = {}
    for sym in close_unstacked.columns:
        code = sym.split(".")[0] if "." in str(sym) else str(sym)
        code = code.upper().lstrip("SH").lstrip("SZ")
        prefix = code[:3]
        sector_groups.setdefault(f"sector_{prefix}", []).append(sym)

    # Keep only groups with ≥ 5 stocks
    sector_ret = pd.DataFrame(index=close_unstacked.index)
    for name, syms in sector_groups.items():
        if len(syms) >= 5:
            grp = close_unstacked[syms]
            sector_ret[name] = grp.pct_change(20).mean(axis=1)

    sector_ret = sector_ret.reindex(cal_idx)
    return sector_ret


def compute_lagged_factor_ic(
    alpha158_df: pd.DataFrame,
    returns: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute rolling IC (Spearman rank correlation) per factor category.

    Args:
        alpha158_df: Alpha158 features, MultiIndex (datetime, instrument).
        returns: Forward returns, same index as alpha158_df.
        lookback: Rolling window in trading days.

    Returns:
        DataFrame indexed by datetime with one column per factor category.
    """
    feature_groups = group_features_by_category(list(alpha158_df.columns))
    dates = sorted(alpha158_df.index.get_level_values("datetime").unique())

    ic_records = []
    for dt in dates:
        row = {"datetime": dt}
        try:
            day_factors = alpha158_df.xs(dt, level="datetime")
            day_ret = returns.xs(dt, level="datetime") if isinstance(returns.index, pd.MultiIndex) else returns.loc[dt]
        except KeyError:
            ic_records.append(row)
            continue

        for cat in FACTOR_CATEGORIES:
            feats = feature_groups.get(cat, [])
            if not feats or day_ret.empty:
                row[f"ic_{cat}"] = 0.0
                continue
            # Mean factor value per stock for this category
            cat_score = day_factors[feats].mean(axis=1)
            # Spearman rank correlation
            valid = pd.DataFrame({"score": cat_score, "ret": day_ret}).dropna()
            if len(valid) < 10:
                row[f"ic_{cat}"] = 0.0
            else:
                row[f"ic_{cat}"] = valid["score"].corr(valid["ret"], method="spearman")
        ic_records.append(row)

    ic_df = pd.DataFrame(ic_records).set_index("datetime")

    # Rolling mean over lookback window
    ic_df = ic_df.rolling(lookback, min_periods=5).mean().fillna(0)
    return ic_df


# ── Raw OHLCV (for Kronos input) ────────────────────────────────────────────

def load_raw_ohlcv(
    symbols: list[str],
    start: str,
    end: str,
    cfg: ScreenerConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Load raw OHLCV data for specific symbols (Kronos input).

    Follows the same pattern as ``finetune/qlib_data_preprocess.py:30-74``.

    Returns:
        Dict mapping symbol → DataFrame with columns
        [open, high, low, close, volume, amount].
    """
    cfg = cfg or ScreenerConfig()
    data_fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]

    loader = QlibDataLoader(config=data_fields)
    raw = loader.load(symbols, start, end)
    raw = raw.stack().unstack(level=1)

    result = {}
    for symbol in symbols:
        if symbol not in raw.columns.get_level_values(0):
            continue
        sdf = raw[symbol]
        sdf = sdf.reset_index().rename(columns={"level_1": "field"})
        sdf = pd.pivot(sdf, index="datetime", columns="field", values=symbol)
        sdf = sdf.rename(columns={
            "$open": "open", "$high": "high", "$low": "low",
            "$close": "close", "$volume": "volume", "$vwap": "vwap",
        })
        sdf["vol"] = sdf["volume"]
        sdf["amt"] = sdf[["open", "high", "low", "close"]].mean(axis=1) * sdf["vol"]
        sdf = sdf[["open", "high", "low", "close", "vol", "amt"]].dropna()
        if len(sdf) >= cfg.kronos_lookback + cfg.kronos_pred_len:
            result[symbol] = sdf
    return result

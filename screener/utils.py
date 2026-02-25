"""
Shared helpers: calendar features, factor category mapping, sector classification.
"""

import numpy as np
import pandas as pd

# ── Calendar Features ────────────────────────────────────────────────────────

def calendar_features(dt: pd.Timestamp) -> dict:
    """Extract calendar features from a single date."""
    quarter = (dt.month - 1) // 3 + 1
    quarter_end_month = quarter * 3
    quarter_end = pd.Timestamp(dt.year, quarter_end_month, 1) + pd.offsets.MonthEnd(0)
    days_to_qend = (quarter_end - dt).days

    return {
        "month": dt.month,
        "quarter": quarter,
        "day_of_week": dt.dayofweek,
        "days_to_quarter_end": days_to_qend,
        "day_of_month": dt.day,
    }


def calendar_features_series(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Vectorised calendar features for a DatetimeIndex."""
    df = pd.DataFrame(index=dates)
    df["month"] = dates.month
    df["quarter"] = (dates.month - 1) // 3 + 1
    df["day_of_week"] = dates.dayofweek
    df["day_of_month"] = dates.day

    # days_to_quarter_end — vectorised
    qend_month = df["quarter"] * 3
    qend = pd.to_datetime(
        {
            "year": dates.year,
            "month": qend_month.values,
            "day": 1,
        }
    ) + pd.offsets.MonthEnd(0)
    df["days_to_quarter_end"] = (qend - dates).dt.days
    return df


# ── Alpha158 Factor Categories ──────────────────────────────────────────────
# Maps Qlib Alpha158 feature *prefix* to a human-readable category.
# Alpha158 features follow the naming pattern: CATEGORY$col_col_N

FACTOR_CATEGORY_PREFIXES = {
    "KMID": "momentum",
    "KLEN": "momentum",
    "KMID2": "momentum",
    "KUP": "momentum",
    "KUP2": "momentum",
    "KLOW": "momentum",
    "KLOW2": "momentum",
    "KSFT": "momentum",
    "KSFT2": "momentum",
    "ROC": "momentum",
    "MA": "trend",
    "STD": "volatility",
    "BETA": "volatility",
    "RSQR": "volatility",
    "RESI": "volatility",
    "MAX": "price_extreme",
    "MIN": "price_extreme",
    "QTLU": "price_extreme",
    "QTLD": "price_extreme",
    "RANK": "cross_section",
    "RSV": "mean_reversion",
    "IMAX": "price_extreme",
    "IMIN": "price_extreme",
    "IMXD": "price_extreme",
    "CORR": "correlation",
    "CORD": "correlation",
    "CNTP": "volume_price",
    "CNTN": "volume_price",
    "CNTD": "volume_price",
    "SUMP": "volume_price",
    "SUMN": "volume_price",
    "SUMD": "volume_price",
    "VMA": "volume",
    "VSTD": "volume",
    "WVMA": "volume",
    "VSUMP": "volume",
    "VSUMN": "volume",
    "VSUMD": "volume",
}

# All unique category names
FACTOR_CATEGORIES = sorted(set(FACTOR_CATEGORY_PREFIXES.values()))


def feature_to_category(feature_name: str) -> str:
    """Map an Alpha158 feature name to its category.

    Feature names look like ``KMID$close_close_0`` or ``ROC$close_5``.
    We match on the text before the first ``$``.
    """
    prefix = feature_name.split("$")[0] if "$" in feature_name else feature_name
    # Strip trailing digits for robustness (e.g. "KMID2" -> lookup directly)
    return FACTOR_CATEGORY_PREFIXES.get(prefix, "other")


def group_features_by_category(feature_names: list[str]) -> dict[str, list[str]]:
    """Group a list of Alpha158 feature names by category."""
    groups: dict[str, list[str]] = {}
    for f in feature_names:
        cat = feature_to_category(f)
        groups.setdefault(cat, []).append(f)
    return groups


# ── Sector Classification (ShenWan L1) ──────────────────────────────────────
# A-share ShenWan Level 1 sector code prefix mapping.  The full mapping is
# loaded dynamically from Qlib instrument metadata; this is a fallback lookup.

SHENWAN_L1_SECTORS = [
    "农林牧渔", "采掘", "化工", "钢铁", "有色金属", "电子", "家用电器",
    "食品饮料", "纺织服装", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商业贸易", "休闲服务", "综合", "建筑材料", "建筑装饰",
    "电气设备", "国防军工", "计算机", "传媒", "通信", "银行",
    "非银金融", "汽车", "机械设备",
]


def get_board_type(symbol: str) -> str:
    """Determine board type from A-share stock code.

    Returns one of: 'main', 'gem' (创业板), 'star' (科创板).
    """
    code = symbol.split(".")[0] if "." in symbol else symbol
    # Strip exchange prefix if present (e.g. SH600000 -> 600000)
    code = code.lstrip("SHshSZsz")

    if code.startswith("300") or code.startswith("301"):
        return "gem"   # 创业板
    if code.startswith("688") or code.startswith("689"):
        return "star"  # 科创板
    return "main"


def get_limit_threshold(symbol: str, is_ipo_first5: bool = False) -> float:
    """Return the limit-up/down threshold for a symbol."""
    board = get_board_type(symbol)
    if board in ("gem", "star"):
        return 0.30 if is_ipo_first5 else 0.20
    return 0.10


def robust_zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """RobustZScore normalisation: (x - median) / (1.4826 * MAD), clipped."""
    median = series.median()
    mad = (series - median).abs().median()
    scaled = (series - median) / (1.4826 * mad + 1e-9)
    return scaled.clip(-clip, clip)

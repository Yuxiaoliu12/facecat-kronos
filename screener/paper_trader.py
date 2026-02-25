"""
Layer 4: Paper Trading Engine

Full-capital-per-trade paper trader with Kronos-predicted exit rules,
A-share lot rounding, commission fees, and 涨停/跌停 (limit) handling.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig
from screener.utils import get_board_type, get_limit_threshold


@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    predicted_candles: pd.DataFrame  # Kronos predicted OHLCV (pred_len rows)
    hold_days: int = 0


@dataclass
class Trade:
    symbol: str
    action: str         # "buy" or "sell"
    date: pd.Timestamp
    price: float
    shares: int
    commission: float
    pnl: float = 0.0   # for sell trades
    reason: str = ""    # exit reason


class PaperTrader:
    """Paper trading engine with Kronos-predicted exit rules."""

    def __init__(self, cfg: ScreenerConfig | None = None):
        self.cfg = cfg or ScreenerConfig()
        self.cash = self.cfg.initial_capital
        self.position: Position | None = None
        self.trade_log: list[Trade] = []
        self.daily_nav: list[tuple[pd.Timestamp, float]] = []  # (date, NAV)
        self._prev_close: dict[str, float] = {}  # for limit-price checks

    # ── Commission ───────────────────────────────────────────────────────

    def _buy_cost(self, price: float, shares: int) -> float:
        return price * shares * self.cfg.buy_commission

    def _sell_cost(self, price: float, shares: int) -> float:
        comm = price * shares * self.cfg.sell_commission
        tax = price * shares * self.cfg.stamp_tax
        return comm + tax

    # ── Limit-Price Checks ───────────────────────────────────────────────

    def _is_limit_up_open(
        self, symbol: str, open_price: float, prev_close: float
    ) -> bool:
        """Check if stock opens at 涨停 (can't buy)."""
        if prev_close <= 0:
            return False
        threshold = get_limit_threshold(symbol)
        return (open_price / prev_close - 1) >= threshold - 0.001  # small tolerance

    def _is_limit_down_open(
        self, symbol: str, open_price: float, prev_close: float
    ) -> bool:
        """Check if stock opens at 跌停 (can't sell)."""
        if prev_close <= 0:
            return False
        threshold = get_limit_threshold(symbol)
        return (open_price / prev_close - 1) <= -threshold + 0.001

    def _is_yizi_ban(self, row: pd.Series) -> bool:
        """Check for 一字板: open == high == low == close."""
        return (
            abs(row["open"] - row["high"]) < 0.001
            and abs(row["high"] - row["low"]) < 0.001
            and abs(row["low"] - row["close"]) < 0.001
        )

    # ── Buy / Sell ───────────────────────────────────────────────────────

    def buy(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: float,
        predicted_candles: pd.DataFrame,
        prev_close: float | None = None,
    ) -> bool:
        """Attempt to buy a stock with full capital.

        Returns True if trade executed, False if blocked (涨停/insufficient cash).
        """
        if self.position is not None:
            return False  # already holding

        # Limit-up check
        if prev_close is not None and self._is_limit_up_open(symbol, price, prev_close):
            return False

        # Calculate shares (round down to lots of 100)
        max_cost = self.cash
        raw_shares = int(max_cost / price)
        shares = (raw_shares // self.cfg.lot_size) * self.cfg.lot_size
        if shares <= 0:
            return False

        cost = price * shares
        commission = self._buy_cost(price, shares)
        total_cost = cost + commission

        if total_cost > self.cash:
            shares -= self.cfg.lot_size
            if shares <= 0:
                return False
            cost = price * shares
            commission = self._buy_cost(price, shares)
            total_cost = cost + commission

        self.cash -= total_cost
        self.position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_date=date,
            predicted_candles=predicted_candles,
            hold_days=0,
        )
        self.trade_log.append(Trade(
            symbol=symbol, action="buy", date=date, price=price,
            shares=shares, commission=commission, reason="entry",
        ))
        return True

    def sell(
        self,
        date: pd.Timestamp,
        price: float,
        reason: str = "",
        prev_close: float | None = None,
    ) -> bool:
        """Attempt to sell current position.

        Returns True if executed, False if blocked (跌停).
        """
        if self.position is None:
            return False

        # Limit-down check
        if prev_close is not None and self._is_limit_down_open(
            self.position.symbol, price, prev_close
        ):
            return False

        shares = self.position.shares
        revenue = price * shares
        commission = self._sell_cost(price, shares)
        net_revenue = revenue - commission

        pnl = net_revenue - (self.position.entry_price * shares + self._buy_cost(
            self.position.entry_price, shares
        ))

        self.cash += net_revenue
        self.trade_log.append(Trade(
            symbol=self.position.symbol, action="sell", date=date,
            price=price, shares=shares, commission=commission,
            pnl=pnl, reason=reason,
        ))
        self.position = None
        return True

    # ── Exit Rule Evaluation ─────────────────────────────────────────────

    def _check_exit_rules(
        self, actual_close: float, actual_row: pd.Series
    ) -> str | None:
        """Check Kronos-predicted exit rules against today's actual close.

        Returns exit reason string or None if no exit triggered.
        All exits execute at next-day open.
        """
        pos = self.position
        if pos is None:
            return None

        pred = pos.predicted_candles
        day_idx = pos.hold_days  # 0-indexed

        # Time limit
        if day_idx >= self.cfg.max_hold_days:
            return "time_limit"

        # Check if we have predictions for this day
        if day_idx >= len(pred):
            return "time_limit"

        pred_row = pred.iloc[day_idx]
        pred_high = pred_row.get("high", pred_row.get("$high", float("inf")))
        pred_low = pred_row.get("low", pred_row.get("$low", 0))
        pred_close = pred_row.get("close", pred_row.get("$close", actual_close))

        # Take-profit: actual close > predicted high
        if actual_close > pred_high:
            return "take_profit"

        # Stop-loss: actual close < predicted low
        if actual_close < pred_low:
            return "stop_loss"

        # Trajectory deviation: actual close deviates from predicted close
        # by more than 2× the predicted daily range
        pred_range = abs(pred_high - pred_low)
        if pred_range > 0:
            deviation = abs(actual_close - pred_close)
            if deviation > self.cfg.deviation_multiplier * pred_range:
                return "trajectory_deviation"

        return None

    # ── Daily Update ─────────────────────────────────────────────────────

    def daily_update(
        self,
        date: pd.Timestamp,
        ranked_symbols: list[str],
        ohlcv_today: dict[str, pd.Series],
        ohlcv_prev: dict[str, pd.Series] | None = None,
        kronos_predictions: dict[str, pd.DataFrame] | None = None,
    ):
        """Process one trading day.

        1. For held positions: check exit rules against yesterday's close,
           execute sell at today's open if triggered.
        2. If no position: buy the top-ranked stock at today's open.

        Args:
            date: Today's date.
            ranked_symbols: Symbols ranked by the pipeline (best first).
            ohlcv_today: symbol → Series with open/high/low/close/volume for today.
            ohlcv_prev: symbol → Series for previous day (for limit checks).
            kronos_predictions: symbol → predicted candles DataFrame.
        """
        prev_close_map = {}
        if ohlcv_prev:
            for sym, row in ohlcv_prev.items():
                prev_close_map[sym] = row.get("close", 0)

        # ── Step 1: Check exits for held position ────────────────────────
        if self.position is not None:
            sym = self.position.symbol
            today_row = ohlcv_today.get(sym)

            if today_row is not None:
                # Exit rules were evaluated yesterday at close → execute at today's open
                exit_reason = self._pending_exit_reason
                if exit_reason:
                    open_price = today_row.get("open", today_row.get("$open", 0))
                    prev_cl = prev_close_map.get(sym, 0)

                    # Check 一字板 / 跌停
                    if self._is_yizi_ban(today_row):
                        pass  # can't sell, carry to next day
                    else:
                        sold = self.sell(date, open_price, exit_reason, prev_cl)
                        if sold:
                            self._pending_exit_reason = None

                # Evaluate today's exit rules (will execute tomorrow)
                actual_close = today_row.get("close", today_row.get("$close", 0))
                self.position.hold_days += 1
                self._pending_exit_reason = self._check_exit_rules(
                    actual_close, today_row
                )
            else:
                # No data for held stock today — force sell next opportunity
                self._pending_exit_reason = "no_data"

        # ── Step 2: Buy if no position ───────────────────────────────────
        if self.position is None and ranked_symbols:
            for sym in ranked_symbols:
                today_row = ohlcv_today.get(sym)
                if today_row is None:
                    continue

                open_price = today_row.get("open", today_row.get("$open", 0))
                if open_price <= 0:
                    continue

                prev_cl = prev_close_map.get(sym, 0)

                # Check 一字板 at 涨停
                if self._is_yizi_ban(today_row) and self._is_limit_up_open(
                    sym, open_price, prev_cl
                ):
                    continue

                # Get Kronos predictions for this stock
                pred_candles = pd.DataFrame()
                if kronos_predictions and sym in kronos_predictions:
                    pred_candles = kronos_predictions[sym]

                bought = self.buy(date, sym, open_price, pred_candles, prev_cl)
                if bought:
                    self._pending_exit_reason = None
                    break

        # ── Record NAV ───────────────────────────────────────────────────
        nav = self.cash
        if self.position is not None:
            sym = self.position.symbol
            today_row = ohlcv_today.get(sym)
            if today_row is not None:
                close = today_row.get("close", today_row.get("$close", 0))
                nav += close * self.position.shares
        self.daily_nav.append((date, nav))

    # ── Metrics ──────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Compute performance metrics from trade log and NAV history."""
        trades = self.trade_log
        sells = [t for t in trades if t.action == "sell"]

        if not sells:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "trade_count": 0,
                "total_pnl": 0.0,
                "total_commission": sum(t.commission for t in trades),
            }

        wins = sum(1 for t in sells if t.pnl > 0)
        total_pnl = sum(t.pnl for t in sells)
        total_comm = sum(t.commission for t in trades)

        # NAV-based metrics
        nav_series = pd.Series(
            {d: v for d, v in self.daily_nav},
            name="nav",
        ).sort_index()

        if len(nav_series) < 2:
            return {
                "total_return": total_pnl / self.cfg.initial_capital,
                "win_rate": wins / len(sells),
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "trade_count": len(sells),
                "total_pnl": total_pnl,
                "total_commission": total_comm,
            }

        daily_returns = nav_series.pct_change().dropna()
        total_return = nav_series.iloc[-1] / self.cfg.initial_capital - 1

        # Annualised Sharpe (252 trading days)
        if daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_dd = drawdown.min()

        return {
            "total_return": float(total_return),
            "win_rate": wins / len(sells),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "trade_count": len(sells),
            "total_pnl": float(total_pnl),
            "total_commission": float(total_comm),
        }

    def get_nav_series(self) -> pd.Series:
        """Return daily NAV as a Series."""
        return pd.Series(
            {d: v for d, v in self.daily_nav}, name="nav"
        ).sort_index()

    def reset(self):
        """Reset the trader to initial state."""
        self.cash = self.cfg.initial_capital
        self.position = None
        self.trade_log = []
        self.daily_nav = []
        self._pending_exit_reason = None
        self._prev_close = {}

    # Initialise mutable internal state
    _pending_exit_reason: str | None = None

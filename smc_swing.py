"""
SMC Swing – Smart Money Concepts Signal Engine (Python Port)
============================================================
Ported from Pine Script v5: "Swing - Smart Money Concepts [LuxAlgo] +
Signals & TP/SL (Swing Optimized)".

Dependencies:
    pip install pandas pandas_ta yfinance

Usage:
    import pandas as pd
    from smc_swing import SMCSwing

    df = pd.read_csv("your_ohlcv.csv", parse_dates=["datetime"], index_col="datetime")
    smc = SMCSwing(df)
    signals = smc.run()
    print(signals[signals["signal"] != 0])
"""

import math
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("Install pandas_ta:  pip install pandas_ta")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BULLISH     =  1
BEARISH     = -1
BULLISH_LEG =  1
BEARISH_LEG =  0

BOS   = "BOS"
CHOCH = "CHoCH"
ALL   = "All"


# ─────────────────────────────────────────────────────────────────────────────
# Config  (mirrors Pine Script inputs)
# ─────────────────────────────────────────────────────────────────────────────
class SMCConfig:
    swing_length:   int   = 50      # "Swing Length"
    internal_length: int  = 20      # "Internal Length"

    # Signal filters
    show_signals:       bool  = True
    risk_reward_ratio:  float = 4.0
    sl_atr_mult:        float = 3.0
    use_vwap_filter:    bool  = False
    use_htf_bias:       bool  = True   # simplified: uses EMA slope as proxy
    min_bars_between:   int   = 10

    # Order-block filter
    ob_filter:       str   = "Atr"           # "Atr" | "Cumulative Mean Range"
    ob_mitigation:   str   = "High/Low"      # "Close" | "High/Low"
    swing_ob_count:  int   = 4
    internal_ob_count: int = 4

    # Equal H/L
    eq_hl_length:    int   = 3
    eq_hl_threshold: float = 0.1

    # FVG
    show_fvg: bool = True

    # Premium / Discount zones
    show_zones: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Helper dataclasses
# ─────────────────────────────────────────────────────────────────────────────
class Pivot:
    def __init__(self):
        self.current_level: float = float("nan")
        self.last_level:    float = float("nan")
        self.crossed:       bool  = False
        self.bar_index:     int   = -1

class Trend:
    def __init__(self):
        self.bias: int = 0   # BULLISH (+1) | BEARISH (-1) | 0

class TrailingExtremes:
    def __init__(self):
        self.top:              float = float("nan")
        self.bottom:           float = float("nan")
        self.bar_index:        int   = -1
        self.last_top_idx:     int   = -1
        self.last_bottom_idx:  int   = -1

class OrderBlock:
    def __init__(self, bar_high, bar_low, bar_idx, bias):
        self.bar_high:  float = bar_high
        self.bar_low:   float = bar_low
        self.bar_index: int   = bar_idx
        self.bias:      int   = bias   # BULLISH | BEARISH

class Alerts:
    def __init__(self):
        self.swing_bullish_bos    = False
        self.swing_bearish_bos    = False
        self.swing_bullish_choch  = False
        self.swing_bearish_choch  = False
        self.internal_bullish_bos = False
        self.internal_bearish_bos = False
        self.internal_bullish_choch = False
        self.internal_bearish_choch = False
        self.equal_highs = False
        self.equal_lows  = False
        self.bullish_fvg = False
        self.bearish_fvg = False

    def reset(self):
        for k in self.__dict__:
            setattr(self, k, False)


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────
class SMCSwing:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume (case-insensitive).
        Index should be a DatetimeIndex.
    cfg : SMCConfig, optional
    """

    def __init__(self, df: pd.DataFrame, cfg: SMCConfig | None = None):
        self.cfg = cfg or SMCConfig()
        self.df  = self._normalise(df)

    # ── public ──────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Iterate bar-by-bar and return a DataFrame with columns:
            signal      : +1 (buy), -1 (sell), 0 (none)
            entry       : entry price (close at signal bar)
            sl          : stop-loss price
            tp          : take-profit price
            structure   : label e.g. "BOS", "CHoCH", ""
            swing_bias  : current swing trend bias (+1 / -1 / 0)
            position    : current open position (+1 / -1 / 0)
        """
        df  = self.df
        cfg = self.cfg
        n   = len(df)

        # Pre-compute indicators (vectorised)
        atr14   = df.ta.atr(length=14)
        ema20   = df.ta.ema(length=20)
        ema50   = df.ta.ema(length=50)
        ema200  = df.ta.ema(length=200)
        vwap    = self._daily_vwap(df)

        # State
        swing_high    = Pivot()
        swing_low     = Pivot()
        internal_high = Pivot()
        internal_low  = Pivot()
        swing_trend   = Trend()
        internal_trend = Trend()
        trailing       = TrailingExtremes()

        swing_obs:    list[OrderBlock] = []
        internal_obs: list[OrderBlock] = []

        position      = 0
        current_sl    = float("nan")
        current_tp    = float("nan")
        last_signal_dir  = 0
        last_signal_bar  = -999

        # Leg state trackers
        swing_leg_val   = BEARISH_LEG
        internal_leg_val = BEARISH_LEG

        # Output accumulators
        out_signal    = np.zeros(n, dtype=int)
        out_entry     = np.full(n, float("nan"))
        out_sl        = np.full(n, float("nan"))
        out_tp        = np.full(n, float("nan"))
        out_structure = [""] * n
        out_bias      = np.zeros(n, dtype=int)
        out_pos       = np.zeros(n, dtype=int)

        for i in range(n):
            if i < 210:
                continue
            high_i  = df["high"].iat[i]
            low_i   = df["low"].iat[i]
            open_i  = df["open"].iat[i]
            close_i = df["close"].iat[i]
            atr_i   = atr14.iat[i]

            alerts = Alerts()

            # ── Swing leg detection ──────────────────────────────────────────
            swing_leg_val = self._update_leg(
                df, i, cfg.swing_length, swing_leg_val,
                swing_high, swing_low, swing_trend, trailing,
                alerts, internal_trend, atr_i,
                swing_obs, internal_obs, cfg,
                ema50, ema200, internal=False
            )

            # ── Internal leg detection ───────────────────────────────────────
            internal_leg_val = self._update_leg(
                df, i, cfg.internal_length, internal_leg_val,
                internal_high, internal_low, internal_trend, trailing,
                alerts, swing_trend, atr_i,
                swing_obs, internal_obs, cfg,
                ema50, ema200, internal=True
            )

            # ── Trailing extremes ────────────────────────────────────────────
            if math.isnan(trailing.top) or high_i > trailing.top:
                trailing.top         = high_i
                trailing.last_top_idx = i
            if math.isnan(trailing.bottom) or low_i < trailing.bottom:
                trailing.bottom          = low_i
                trailing.last_bottom_idx = i
            trailing.bar_index = i

            # ── Order block mitigation ───────────────────────────────────────
            mit_high = close_i if cfg.ob_mitigation == "Close" else high_i
            mit_low  = close_i if cfg.ob_mitigation == "Close" else low_i
            swing_obs    = [ob for ob in swing_obs
                            if not (mit_high > ob.bar_high and ob.bias == BEARISH)
                            and not (mit_low  < ob.bar_low  and ob.bias == BULLISH)]
            internal_obs = [ob for ob in internal_obs
                            if not (mit_high > ob.bar_high and ob.bias == BEARISH)
                            and not (mit_low  < ob.bar_low  and ob.bias == BULLISH)]

            # ── Raw signals ──────────────────────────────────────────────────
            raw_buy  = alerts.swing_bullish_bos  or alerts.swing_bullish_choch
            raw_sell = alerts.swing_bearish_bos  or alerts.swing_bearish_choch

            # ── Filters ──────────────────────────────────────────────────────
            a50  = ema50.iat[i]   if i < len(ema50)  else float("nan")
            a200 = ema200.iat[i]  if i < len(ema200) else float("nan")
            trend_bull = close_i > a50 and a50 > a200
            trend_bear = close_i < a50 and a50 < a200

            e20 = ema20.iat[i] if i < len(ema20) else float("nan")
            pullback_buy  = low_i  < e20
            pullback_sell = high_i > e20

            vwap_ok_buy  = (not cfg.use_vwap_filter) or (close_i > vwap.iat[i])
            vwap_ok_sell = (not cfg.use_vwap_filter) or (close_i < vwap.iat[i])

            candle_move = abs(high_i - low_i) >= atr_i * 0.6

            bars_ok = (i - last_signal_bar) > cfg.min_bars_between

            buy_signal = (
                cfg.show_signals
                and raw_buy
                and candle_move
                and vwap_ok_buy
                and trend_bull
                and position == 0
                and last_signal_dir != 1
                and pullback_buy
                and bars_ok
            )

            sell_signal = (
                cfg.show_signals
                and raw_sell
                and candle_move
                and vwap_ok_sell
                and trend_bear
                and position == 0
                and last_signal_dir != -1
                and pullback_sell
                and bars_ok
            )

            # ── Exit logic ───────────────────────────────────────────────────
            sl_hit_long  = position ==  1 and not math.isnan(current_sl) and low_i  <= current_sl
            tp_hit_long  = position ==  1 and not math.isnan(current_tp) and high_i >= current_tp
            sl_hit_short = position == -1 and not math.isnan(current_sl) and high_i >= current_sl
            tp_hit_short = position == -1 and not math.isnan(current_tp) and low_i  <= current_tp

            exit_long  = position ==  1 and (sl_hit_long  or tp_hit_long  or raw_sell)
            exit_short = position == -1 and (sl_hit_short or tp_hit_short or raw_buy)

            if exit_long or exit_short:
                position = 0

            # ── Entry & SL/TP calculation ────────────────────────────────────
            if buy_signal or sell_signal:
                entry      = close_i
                lookback   = 5
                start      = max(0, i - lookback + 1)
                recent_low  = df["low"].iloc[start:i+1].min()
                recent_high = df["high"].iloc[start:i+1].max()
                buffer     = atr_i * 0.1
                max_risk   = atr_i * 2.5

                if buy_signal:
                    raw_sl  = recent_low - buffer
                    capped  = entry - max_risk
                    sl_price = max(raw_sl, capped)
                    risk     = entry - sl_price
                    tp_price = entry + risk * 2

                    position       = 1
                    last_signal_dir = 1
                    last_signal_bar = i

                    out_signal[i] = 1
                    out_entry[i]  = entry
                    out_sl[i]     = sl_price
                    out_tp[i]     = tp_price
                    struct_tag = (BOS if alerts.swing_bullish_bos else CHOCH)
                    out_structure[i] = f"BULL {struct_tag}"

                else:  # sell_signal
                    raw_sl   = recent_high + buffer
                    capped   = entry + max_risk
                    sl_price = min(raw_sl, capped)
                    risk     = sl_price - entry
                    tp_price = entry - risk * 2

                    position       = -1
                    last_signal_dir = -1
                    last_signal_bar = i

                    out_signal[i] = -1
                    out_entry[i]  = entry
                    out_sl[i]     = sl_price
                    out_tp[i]     = tp_price
                    struct_tag = (BOS if alerts.swing_bearish_bos else CHOCH)
                    out_structure[i] = f"BEAR {struct_tag}"

                current_sl = sl_price
                current_tp = tp_price

            out_bias[i] = swing_trend.bias
            out_pos[i]  = position

        return pd.DataFrame({
            "signal":    out_signal,
            "entry":     out_entry,
            "sl":        out_sl,
            "tp":        out_tp,
            "structure": out_structure,
            "swing_bias": out_bias,
            "position":  out_pos,
        }, index=df.index)

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"DataFrame missing column: {col}")
        return df

    @staticmethod
    def _daily_vwap(df: pd.DataFrame) -> pd.Series:
        """Rolling VWAP that resets each calendar day."""
        hlc3   = (df["high"] + df["low"] + df["close"]) / 3
        vol    = df.get("volume", pd.Series(np.ones(len(df)), index=df.index))
        dates  = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        result = pd.Series(float("nan"), index=df.index)
        for day, grp in df.groupby(dates):
            idx  = grp.index
            tp   = hlc3.loc[idx]
            v    = vol.loc[idx]
            cum_tpv = (tp * v).cumsum()
            cum_v   = v.cumsum()
            result.loc[idx] = cum_tpv / cum_v
        return result

    def _leg(self, df: pd.DataFrame, i: int, size: int) -> int:
        """
        Compute leg direction at bar i.
        Returns BEARISH_LEG (0) if high[size bars ago] > rolling max of last `size` bars,
        BULLISH_LEG (1) if low[size bars ago]  < rolling min of last `size` bars.
        """
        if i < size:
            return BEARISH_LEG
        pivot_bar = i - size
        window_high = df["high"].iloc[max(0, i - size + 1): i + 1].max()
        window_low  = df["low"].iloc[max(0, i - size + 1): i + 1].min()
        if df["high"].iat[pivot_bar] > window_high:
            return BEARISH_LEG
        if df["low"].iat[pivot_bar] < window_low:
            return BULLISH_LEG
        return BEARISH_LEG   # default (unchanged)

    def _update_leg(
        self, df, i, size, prev_leg,
        p_high: Pivot, p_low: Pivot,
        trend: Trend, trailing: TrailingExtremes,
        alerts: Alerts, other_trend: Trend,
        atr_i: float,
        swing_obs, internal_obs,
        cfg: SMCConfig,
        ema50, ema200,
        internal: bool
    ) -> int:
        """
        Mirror of Pine Script's leg() + getCurrentStructure() combined.
        Returns updated leg value.
        """
        cur_leg = self._leg(df, i, size)
        if cur_leg == prev_leg:
            # No new pivot yet – still check for structure crosses
            self._check_structure_cross(
                df, i, p_high, p_low, trend, alerts,
                swing_obs, internal_obs, cfg, internal
            )
            return cur_leg

        # ── New pivot formed ──────────────────────────────────────────────
        pivot_bar = i - size
        if pivot_bar < 0:
            return cur_leg

        if cur_leg == BULLISH_LEG:          # new swing low
            p_low.last_level    = p_low.current_level
            p_low.current_level = df["low"].iat[pivot_bar]
            p_low.crossed       = False
            p_low.bar_index     = pivot_bar
            if not internal:
                trailing.bottom          = p_low.current_level
                trailing.last_bottom_idx = pivot_bar
                trailing.bar_index       = pivot_bar
        else:                               # new swing high
            p_high.last_level    = p_high.current_level
            p_high.current_level = df["high"].iat[pivot_bar]
            p_high.crossed       = False
            p_high.bar_index     = pivot_bar
            if not internal:
                trailing.top          = p_high.current_level
                trailing.last_top_idx = pivot_bar
                trailing.bar_index    = pivot_bar

        self._check_structure_cross(
            df, i, p_high, p_low, trend, alerts,
            swing_obs, internal_obs, cfg, internal
        )
        return cur_leg

    def _check_structure_cross(
        self, df, i, p_high: Pivot, p_low: Pivot,
        trend: Trend, alerts: Alerts,
        swing_obs, internal_obs,
        cfg: SMCConfig, internal: bool
    ):
        """BOS / CHoCH detection (crossover / crossunder of pivot levels)."""
        if i < 1:
            return
        close_prev = df["close"].iat[i - 1]
        close_now  = df["close"].iat[i]

        # ── Bullish cross of swing high ───────────────────────────────────
        lvl_h = p_high.current_level
        if (not math.isnan(lvl_h)
                and not p_high.crossed
                and close_prev <= lvl_h < close_now):
            tag = CHOCH if trend.bias == BEARISH else BOS
            if internal:
                if tag == CHOCH:
                    alerts.internal_bullish_choch = True
                else:
                    alerts.internal_bullish_bos = True
            else:
                if tag == CHOCH:
                    alerts.swing_bullish_choch = True
                else:
                    alerts.swing_bullish_bos = True
            p_high.crossed = True
            trend.bias = BULLISH
            # Store order block
            self._store_order_block(
                df, i, p_high, internal, BULLISH, swing_obs, internal_obs, cfg
            )

        # ── Bearish cross of swing low ────────────────────────────────────
        lvl_l = p_low.current_level
        if (not math.isnan(lvl_l)
                and not p_low.crossed
                and close_prev >= lvl_l > close_now):
            tag = CHOCH if trend.bias == BULLISH else BOS
            if internal:
                if tag == CHOCH:
                    alerts.internal_bearish_choch = True
                else:
                    alerts.internal_bearish_bos = True
            else:
                if tag == CHOCH:
                    alerts.swing_bearish_choch = True
                else:
                    alerts.swing_bearish_bos = True
            p_low.crossed = True
            trend.bias = BEARISH
            self._store_order_block(
                df, i, p_low, internal, BEARISH, swing_obs, internal_obs, cfg
            )

    def _store_order_block(
        self, df, i, p: Pivot, internal: bool, bias: int,
        swing_obs, internal_obs, cfg: SMCConfig
    ):
        """Find the candle with highest high (bearish OB) or lowest low (bullish OB)
        between the pivot bar and current bar, and store it."""
        start = p.bar_index
        end   = i
        if start >= end:
            return
        highs = df["high"].iloc[start:end]
        lows  = df["low"].iloc[start:end]
        if bias == BEARISH:
            idx = int(highs.values.argmax()) + start
        else:
            idx = int(lows.values.argmin()) + start

        ob = OrderBlock(
            bar_high  = df["high"].iat[idx],
            bar_low   = df["low"].iat[idx],
            bar_idx   = idx,
            bias      = bias
        )
        obs = internal_obs if internal else swing_obs
        max_count = cfg.internal_ob_count if internal else cfg.swing_ob_count
        obs.insert(0, ob)
        if len(obs) > max_count:
            obs.pop()

    def get_last_signal(self):
        result = self.run()
        return result.iloc[-1]
import yfinance as yf
import pandas as pd
import requests
import os
import time
from io import StringIO
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# CONFIG
# ==============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

INTERVAL = "1d"
PERIOD = "6mo"
MAX_WORKERS = 10
MAX_ALERTS = 10
LTP_MAX = 500

SEEN_FILE = "/tmp/seen_signals.txt"
STOCKS_CACHE_FILE = "/tmp/nifty500_stocks.txt"
STOCKS_CACHE_HOURS = 24

# ==============================
# FETCH NIFTY 500 STOCKS
# ==============================
def fallback_stocks():
    return [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "SBIN.NS", "LT.NS", "ITC.NS",
        "AXISBANK.NS", "KOTAKBANK.NS"
    ]

def get_nifty500_stocks():
    # Use cache if fresh
    if os.path.exists(STOCKS_CACHE_FILE):
        age_hours = (time.time() - os.path.getmtime(STOCKS_CACHE_FILE)) / 3600
        if age_hours < STOCKS_CACHE_HOURS:
            with open(STOCKS_CACHE_FILE) as f:
                symbols = f.read().splitlines()
            print(f"Loaded {len(symbols)} stocks from cache")
            return symbols

    # Fetch fresh from NSE
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.nseindia.com/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        symbols = [s.strip() + ".NS" for s in df["Symbol"].tolist()]

        # Save to cache
        with open(STOCKS_CACHE_FILE, "w") as f:
            f.write("\n".join(symbols))

        print(f"Fetched and cached {len(symbols)} stocks from NSE")
        return symbols
    except Exception as e:
        print(f"Failed to fetch Nifty 500: {e}")
        return fallback_stocks()

def calculate_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr.iloc[-1]

# ==============================
# TELEGRAM
# ==============================
def send_alert(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        })
        print("Telegram status:", response.status_code)
    except Exception as e:
        print("Telegram error:", e)

# ==============================
# MARKET HOURS FILTER (India)
# ==============================
def is_market_open():
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    hour = now.hour + now.minute / 60
    return 9.25 <= hour <= 15.5  # 9:15 AM to 3:30 PM IST

# ==============================
# DUPLICATE FILTER
# ==============================
def load_seen():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r") as f:
            return set(f.read().splitlines())
    return set()

def save_seen(seen):
    with open(SEEN_FILE, "w") as f:
        f.write("\n".join(seen))

# ==============================
# SWING SCORING ENGINE
# ==============================
def score_swing(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- EMA Trend ---
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()

    trend_up = (ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1])
    price_above_ema20 = close.iloc[-1] > ema20.iloc[-1]

    # --- RSI ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1]
    rsi_ok = 40 <= rsi_val <= 65

    # --- Volume Surge ---
    avg_vol = volume.rolling(20).mean().iloc[-1]
    last_vol = volume.iloc[-1]
    volume_surge = last_vol > 1.5 * avg_vol

    # --- Pullback to EMA20 ---
    ltp = close.iloc[-1]
    near_ema20 = abs(ltp - ema20.iloc[-1]) / ema20.iloc[-1] < 0.03

    # --- 52-week high breakout ---
    week52_high = high.rolling(252).max().iloc[-1]
    near_breakout = ltp >= 0.95 * week52_high

    # --- Higher highs & higher lows (last 5 candles) ---
    recent_highs = high.iloc[-5:]
    recent_lows = low.iloc[-5:]
    hh_hl = (
        recent_highs.iloc[-1] > recent_highs.iloc[0] and
        recent_lows.iloc[-1] > recent_lows.iloc[0]
    )

    # --- Scoring ---
    score = 0
    if trend_up:            score += 30
    if price_above_ema20:   score += 10
    if rsi_ok:              score += 20
    if volume_surge:        score += 15
    if near_ema20:          score += 10
    if near_breakout:       score += 10
    if hh_hl:               score += 5

    # --- Signal Type ---
    if near_breakout and trend_up and volume_surge:
        signal_type = "📈 Breakout Setup"
    elif near_ema20 and trend_up and rsi_ok:
        signal_type = "🔄 Pullback to EMA"
    elif trend_up and hh_hl:
        signal_type = "🚀 Momentum Swing"
    else:
        signal_type = None

    return score, signal_type, round(rsi_val, 1)

# ==============================
# SUPPORT / RESISTANCE LEVELS
# ==============================
def get_sr_levels(df):
    recent = df.tail(10)  # tighter for swing
    support = round(recent["low"].min(), 2)
    resistance = round(recent["high"].max(), 2)
    return support, resistance

# ==============================
# PROCESS SINGLE STOCK
# ==============================
def process_stock(stock, seen_signals):
    try:
        time.sleep(0.1)  # avoid yfinance rate limiting

        df = yf.download(
            stock,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=True
        )

        if df.empty or len(df) < 60:
            return None

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        ltp = round(df["close"].iloc[-1], 2)

        # LTP filter
        if ltp > LTP_MAX:
            return None

        score, signal_type, rsi = score_swing(df)

        if score < 60 or signal_type is None:
            return None

        key = f"{stock}_{signal_type}"
        if key in seen_signals:
            return None

        support, resistance = get_sr_levels(df)
        atr = calculate_atr(df)

        # --- Hybrid SL (ATR + Structure) ---
        sl_atr = ltp - 1.2 * atr
        sl_structure = support * 0.995

        sl = round(max(sl_atr, sl_structure), 2)

        # --- Ensure SL is not too wide (max 8%) ---
        max_sl = ltp * 0.92
        sl = max(sl, round(max_sl, 2))

        # --- TP with resistance cap ---
        tp_rr = ltp + (ltp - sl) * 2
        tp = round(min(tp_rr, resistance), 2)

        message = (
            f"<b>{signal_type}: {stock.replace('.NS', '')}</b>\n"
            f"LTP:        ₹{ltp}\n"
            f"Score:      {score}/100\n"
            f"RSI:        {rsi}\n"
            f"Support:    ₹{support}\n"
            f"Resistance: ₹{resistance}\n"
            f"SL:         ₹{sl}\n"
            f"TP:         ₹{tp} (1:2 RR)\n"
        )

        return score, key, message

    except Exception as e:
        print(f"{stock} error: {e}")
        return None

# ==============================
# MAIN SCANNER
# ==============================
def run_scanner():
    IST = timezone(timedelta(hours=5, minutes=30))
    print("Scanner started at", datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"))

    stocks = get_nifty500_stocks()
    seen_signals = load_seen()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_stock, stock, seen_signals): stock for stock in stocks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Sort by score — best picks first
    results.sort(key=lambda x: x[0], reverse=True)
    top_picks = results[:MAX_ALERTS]

    if top_picks:
        new_keys = [r[1] for r in top_picks]
        messages = [r[2] for r in top_picks]

        seen_signals.update(new_keys)
        save_seen(seen_signals)

        header = (
            f"<b>🎯 Top {len(top_picks)} Swing Picks (LTP ≤ ₹500)</b>\n"
            f"<i>{datetime.now(IST).strftime('%d %b %Y %H:%M IST')}</i>\n\n"
        )

        send_alert(header + "\n\n".join(messages))
        print(f"{len(top_picks)} swing picks sent")
    else:
        print("No swing setups found today")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    run_scanner()
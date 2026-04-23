import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime

# ==============================
# CONFIG
# ==============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Start small → scale later
STOCKS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS",
    "ICICIBANK.NS","SBIN.NS","LT.NS","ITC.NS"
]

INTERVAL = "1h"
PERIOD = "3mo"

# ==============================
# TELEGRAM
# ==============================
def send_alert(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
    except:
        pass


# ==============================
# MARKET HOURS FILTER (India)
# ==============================
def is_market_open():
    now = datetime.utcnow()
    hour = now.hour + 5.5  # IST adjust

    return 9 <= hour <= 15.5


# ==============================
# LOAD YOUR SMC ENGINE
# ==============================
from smc_swing import SMCSwing   # your file


# ==============================
# MAIN SCANNER
# ==============================
def run_scanner():

    if not is_market_open():
        print("Market closed. Skipping scan.")
        return

    alerts = []

    for stock in STOCKS:
        try:
            df = yf.download(stock, period=PERIOD, interval=INTERVAL, progress=False)

            if df.empty or len(df) < 100:
                continue

            df.columns = [c.lower() for c in df.columns]

            smc = SMCSwing(df)
            result = smc.run()

            last = result.iloc[-1]

            if last["signal"] == 1:
                alerts.append(
                    f"🚀 BUY: {stock}\nEntry: {last['entry']:.2f}\nSL: {last['sl']:.2f}\nTP: {last['tp']:.2f}"
                )

            elif last["signal"] == -1:
                alerts.append(
                    f"🔻 SELL: {stock}\nEntry: {last['entry']:.2f}\nSL: {last['sl']:.2f}\nTP: {last['tp']:.2f}"
                )

        except Exception as e:
            print(f"{stock} error: {e}")

    # Send all alerts together (faster + cleaner)
    if alerts:
        send_alert("\n\n".join(alerts))
    else:
        print("No signals")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    run_scanner()
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
    # 🔍 DEBUG HERE
    print("TOKEN:", TELEGRAM_TOKEN)
    print("CHAT_ID:", CHAT_ID)
    
    send_alert("🚀 TEST FROM GITHUB ACTIONS")
    
    


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    run_scanner()
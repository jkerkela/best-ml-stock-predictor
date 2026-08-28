import time
import argparse
import asyncio
from datetime import datetime, timedelta
import pytz
import requests

from telegram import Bot

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot_common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk

URL = "https://financialmodelingprep.com/stable/earnings-calendar?apikey="
EPS_SURPRISE_THRESHOLD = 20

# Same timezone as the orchestrator uses for scheduling
EASTERN_TZ = pytz.timezone('US/Eastern')

def getEPSItemsFrom(url, api_key):

    today = datetime.now(EASTERN_TZ).date()
    yesterday = today - timedelta(days=1)
    params = {
        "from": yesterday,
        "to": today
    }
    response = requests.get(f"{url}{api_key}", params=params, timeout=30)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch EPS items from URL={url} with return code={response.status_code}")
        return None

                
async def main(args):
    print("Running EPS monitor")
    telegram_bot = Bot(token=args.telegram_api_token)
    eps_items = getEPSItemsFrom(URL, args.fmp_api_key)
    if eps_items:
        notified_items = set()
        try:
            notified_items = loadObjectFromDisk("EPS_NOTIFIED_ITEMS_DISK_FILE")
        except FileNotFoundError:
            pass
        # Entries older than the query window (yesterday-today) can never match again, prune them
        yesterday = datetime.now(EASTERN_TZ).date() - timedelta(days=1)
        notified_items = {k for k in notified_items if k[1] >= str(yesterday)}
        for item in eps_items:
            actual_eps = item["epsActual"]
            estimated_eps = item["epsEstimated"]
            if actual_eps is None or estimated_eps is None:
                continue
            item_key = (item["symbol"], item["date"])
            if item_key in notified_items:
                continue
            eps_diff_abs = actual_eps - estimated_eps
            if estimated_eps != 0:
                eps_surprise_percent = (eps_diff_abs / abs(estimated_eps)) * 100
            else:
                eps_surprise_percent = eps_diff_abs * 100
            if eps_surprise_percent >= EPS_SURPRISE_THRESHOLD:
                message = f"Found company: {item["symbol"]} with EPS surprise of {eps_surprise_percent}%"
                if await postTelegramNotification(message, telegram_bot, args.telegram_notification_group_id):
                    notified_items.add(item_key)
                    saveObjectToDisk(notified_items, "EPS_NOTIFIED_ITEMS_DISK_FILE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("EPS_monitor")
    parser.add_argument('--fmp_api_key', required=True)
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to execute eps monitor with error={e}")

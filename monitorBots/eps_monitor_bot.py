import time
import argparse
import asyncio
from datetime import date, timedelta
import requests

from telegram import Bot

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk

URL = "https://financialmodelingprep.com/stable/earnings-calendar?apikey="
EPS_SURPRISE_THRESHOLD = 30
        
def getEPSItemsFrom(url, api_key):
    
    today = date.today()
    yesterday = today - timedelta(days=1)
    params = {
        "from": yesterday,
        "to": today
    }
    response = requests.get(f"{url}{api_key}", params=params)
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
        previously_stored_eps_object = None
        try:
            previously_stored_eps_object = loadObjectFromDisk(f"EPS_OBJECT_DISK_FILE")
        except: 
            pass
        if previously_stored_eps_object == eps_items:
            return
        else:
            print(f"New items on EPS items, storing to disk")
            saveObjectToDisk(eps_items, f"EPS_OBJECT_DISK_FILE")
        for item in eps_items:
            actual_eps = item["epsActual"]
            estimated_eps = item["epsEstimated"]
            if actual_eps is None or estimated_eps is None:
                continue
            eps_diff_abs = actual_eps - estimated_eps
            if estimated_eps != 0:
                eps_surprise_percent = (eps_diff_abs / abs(estimated_eps)) * 100
            else:
                eps_surprise_percent = eps_diff_abs * 100
            if eps_surprise_percent >= EPS_SURPRISE_THRESHOLD:
                message = f"Found company: {item["symbol"]} with EPS surprise of {eps_surprise_percent}%"
                await postTelegramNotification(message, telegram_bot, args.telegram_notification_group_id)

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

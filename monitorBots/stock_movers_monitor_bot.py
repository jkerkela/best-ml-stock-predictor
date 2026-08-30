import argparse
import asyncio
from datetime import datetime
import pytz

import yfinance as yf
from tradingview_screener import Query, col

from telegram import Bot

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot_common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk

NOTIFIED_ITEMS_DISK_FILE = "MOVERS_NOTIFIED_ITEMS_DISK_FILE"
MARKET_CHANGE_PERCENT_THRESHOLD = 10
MARKET_CHANGE_PERCENT_THRESHOLD_BIG = 20
HUNDRED_MILLION = 100000000

# Same timezone as the orchestrator uses for scheduling
EASTERN_TZ = pytz.timezone('US/Eastern')

def loadNotifiedItems():
    notified_items = set()
    try:
        notified_items = loadObjectFromDisk(NOTIFIED_ITEMS_DISK_FILE)
    except FileNotFoundError:
        pass
    # Flush posted entries from previous days so each ticker can alert again on a new day
    today = str(datetime.now(EASTERN_TZ).date())
    return {k for k in notified_items if k[2] == today}, today

def getMarketMovers(screen_name):
    result = yf.screen(screen_name, count=5)
    return [(item["symbol"], item.get("shortName"), item.get("regularMarketChangePercent"))
            for item in result["quotes"]]

def getExtendedHoursMovers(market_to_check, ascending):
    _, movers_df = (Query()
        .select(market_to_check)
        .where(col('market_cap_basic') > (HUNDRED_MILLION * 5))
        .order_by(market_to_check, ascending=ascending)
        .limit(5)
        .get_scanner_data()
    )
    return [(row.iloc[0], None, row.iloc[1]) for _, row in movers_df.iterrows()]

async def notifyMovers(movers, mover_label, change_label, threshold, notified_items, today, telegram_bot, args):
    for stock_symbol, company_name, market_change_percent in movers:
        if market_change_percent is None or abs(market_change_percent) <= threshold:
            continue
        item_key = (stock_symbol, args.mode, today)
        if item_key in notified_items:
            print(f"Already notified about {stock_symbol} in {args.mode} mode today, skipping")
            continue
        display_name = f"{company_name} ({stock_symbol})" if company_name else stock_symbol
        message = f"Found {mover_label} with large change: {display_name}, {change_label} change: {market_change_percent:.1f}%"
        print("posting the notification")
        if await postTelegramNotification(message, telegram_bot, args.telegram_notification_group_id):
            notified_items.add(item_key)
            saveObjectToDisk(notified_items, NOTIFIED_ITEMS_DISK_FILE)

async def main(args):
    print("Running stock movers monitor")
    telegram_bot = Bot(token=args.telegram_api_token)
    notified_items, today = loadNotifiedItems()
    if args.mode == "market":
        losers = getMarketMovers('day_losers')
        gainers = getMarketMovers('day_gainers')
        change_label = "market"
        gainer_threshold = MARKET_CHANGE_PERCENT_THRESHOLD_BIG
    else:
        market_to_check = "premarket_change" if args.mode == "premarket" else "postmarket_change"
        losers = getExtendedHoursMovers(market_to_check, ascending=True)
        gainers = getExtendedHoursMovers(market_to_check, ascending=False)
        change_label = args.mode
        gainer_threshold = MARKET_CHANGE_PERCENT_THRESHOLD
    await notifyMovers(losers, "loser", change_label, MARKET_CHANGE_PERCENT_THRESHOLD_BIG,
                       notified_items, today, telegram_bot, args)
    await notifyMovers(gainers, "gainer", change_label, gainer_threshold,
                       notified_items, today, telegram_bot, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("stock_movers_monitor")
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    parser.add_argument("--mode", choices=["premarket", "market", "postmarket"], required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to execute stock movers monitor with error={e}")

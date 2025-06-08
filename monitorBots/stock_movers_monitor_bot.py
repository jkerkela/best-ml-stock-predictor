import argparse
import asyncio

import yfinance as yf
from tradingview_screener import Query, col

from telegram import Bot

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_tools import postTelegramNotification

MARKET_CHANGE_PERCENT_THRESHOLD = 10
HUNDRED_MILLION = 100000000
                
async def main(args):
    print("Running stock movers monitor")
    telegram_bot = Bot(token=args.telegram_api_token)
    if args.mode == "market":
        daily_losers_res = yf.screen('day_losers', count=5)
        for item in daily_losers_res["quotes"]:
            company_name = item["shortName"]
            stock_symbol = item["symbol"]
            market_change = item["regularMarketChangePercent"]
            if abs(market_change) > MARKET_CHANGE_PERCENT_THRESHOLD:
                message = f"Found loser with large change: {company_name} ({stock_symbol}), market change: {market_change}%"
                print("posting the notification")
                await postNotification(message, telegram_bot, args.telegram_notification_group_id)
    else:
        market_to_check = "premarket_change" if args.mode == "premarket" else "postmarket_change"
        _, losers_df = premarket_losers = (Query()
            .select(market_to_check)
            .where(col('market_cap_basic') > HUNDRED_MILLION)
            .order_by(market_to_check, ascending=True)
            .limit(5)
            .get_scanner_data()
        )
        for index, loser in losers_df.iterrows():
            company_symbol = loser.iloc[0]
            market_change_percent = loser.iloc[1]
            if abs(market_change_percent) > MARKET_CHANGE_PERCENT_THRESHOLD:
                message = f"Found loser with large change: {company_symbol}, {market_to_check} change: {market_change_percent}%"
                print("posting the notification")
                await postTelegramNotification(message, telegram_bot, args.telegram_notification_group_id)
            
    
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
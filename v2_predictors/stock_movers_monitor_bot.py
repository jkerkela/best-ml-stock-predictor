import argparse
import asyncio

import yfinance as yf
from telegram import Bot

parser = argparse.ArgumentParser("stock_movers_monitor")
parser.add_argument('--telegram_api_token', required=True)
parser.add_argument('--telegram_notification_group_id', required=True)
args = parser.parse_args()

PREMARKET_CHANGE_PERCENT_THRESHOLD = 10
MARKET_CHANGE_PERCENT_THRESHOLD = 10

async def postNotification(message, telegram_bot, notification_group):
    print("Executing the async notification posting loop")
    retries = 3
    for attempt in range(retries):
        try:
            await telegram_bot.send_message(chat_id=notification_group, text=message)
            print(f"Sent message to notification group as {message}")
            return True
        except Exception as e:
            if attempt < retries - 1:
                print(f"DEBUG: message send attempt failed with error: {e}, retrying...")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
            else:
                print(f"DEBUG: message send attempt failed with error: {e}, retry attemps exceeded")
                return False
                
async def main():
    daily_losers_res = yf.screen('day_losers', count=5)
    telegram_bot = Bot(token=args.telegram_api_token)
    for item in daily_losers_res["quotes"]:
        company_name = item["shortName"]
        stock_symbol = item["symbol"]
        premarket_change = item.get("preMarketChangePercent", 0)
        market_change = item["regularMarketChangePercent"]
        if abs(premarket_change) > PREMARKET_CHANGE_PERCENT_THRESHOLD or abs(market_change) > MARKET_CHANGE_PERCENT_THRESHOLD:
            message = f"Found loser with large change: {company_name} ({stock_symbol}), premarket change: {premarket_change}%, market change: {market_change}%"
            print("posting the notification")
            await postNotification(message, telegram_bot, args.telegram_notification_group_id)
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
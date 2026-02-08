import argparse
import asyncio

from playwright.async_api import async_playwright
from telegram import Bot
from typing import TypedDict

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_tools import postTelegramNotification

IV_STATS_URL = "https://optioncharts.io/trending/highest-implied-volatility-stock-tickers"
IV_STATS_TABLE_NAME = "table.table-sm.table-hover"

COMPANY_SYMBOL_COLUMN = 0
COMPANY_NAME_COLUMN = 1
IV_COLUMN_COLUMN = 2
OPEN_INTEREST_COLUMN = 4

IV_LIMIT_IN_PERCENTS = 100
OPEN_INTEREST_LIMIT = 1_500_000

class IVItem(TypedDict):
    company_name: str
    company_symbol: str
    iv_percent: float
    open_interest: int

async def parseIVDataFrom(URL):
    high_iv_items = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f"Reading IV stats from ULR: {URL}")
        await page.goto(URL)
        
        await page.wait_for_selector(IV_STATS_TABLE_NAME)
        
        rows = await page.locator(f"{IV_STATS_TABLE_NAME} tr").all()
        
        for row in rows:
            data = await row.locator("td").all_inner_texts()
            if data and any(field.strip() for field in data):
                open_interest_value = int(data[OPEN_INTEREST_COLUMN].replace(',', ''))
                iv_value_percent = float(data[IV_COLUMN_COLUMN].replace('%', ''))
                if open_interest_value > OPEN_INTEREST_LIMIT and  iv_value_percent > IV_LIMIT_IN_PERCENTS:
                    print(f"Company={data[COMPANY_NAME_COLUMN]} ({data[COMPANY_SYMBOL_COLUMN]}) with high IV={iv_value_percent} with open interest={open_interest_value}")
                    result_item: IVItem = {
                        "company_name": data[COMPANY_NAME_COLUMN],
                        "company_symbol": data[COMPANY_SYMBOL_COLUMN],
                        "iv_percent": iv_value_percent,
                        "open_interest": open_interest_value
                    }
                    high_iv_items.append(result_item)
        await browser.close()
    return high_iv_items
    
async def main(args):
    telegram_bot = Bot(token=args.telegram_api_token)
    high_iv_items = await parseIVDataFrom(IV_STATS_URL)
    for item in high_iv_items:
        message = f"Found company: {item["company_name"]} ({item["company_symbol"]}) with high IV {item["iv_percent"]}% with open interest {item["open_interest"]}"
        await postTelegramNotification(message, telegram_bot, args.telegram_notification_group_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("IV_monitor")
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
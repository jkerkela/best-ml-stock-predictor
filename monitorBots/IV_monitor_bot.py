import argparse
import asyncio

from playwright.async_api import async_playwright
from telegram import Bot
from typing import TypedDict

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot_common_tools import postTelegramNotification

IV_STATS_URL = "https://optioncharts.io/trending/high-iv-rank-all"
IV_STATS_TABLE_NAME = "table.table-sm.table-hover"

COMPANY_SYMBOL_COLUMN = 0
COMPANY_NAME_COLUMN = 1
IV_RANK_COLUMN = 2
IV_30D_COLUMN = 3
OPEN_INTEREST_COLUMN = 5

IV_RANK_LIMIT = 90.0      
OPEN_INTEREST_LIMIT = 1_500_000

class IVItem(TypedDict):
    company_name: str
    company_symbol: str
    iv_rank: float
    iv_30d: float
    open_interest: int

async def parseIVDataFrom(URL):
    high_iv_items = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f"Reading IV Rank stats from URL: {URL}")
        await page.goto(URL)
        
        # Wait for the specific table to load
        await page.wait_for_selector(IV_STATS_TABLE_NAME)
        
        rows = await page.locator(f"{IV_STATS_TABLE_NAME} tr").all()
        
        for row in rows:
            data = await row.locator("td").all_inner_texts()
            # Ensure the row has enough columns and isn't a header/empty
            if data and len(data) > OPEN_INTEREST_COLUMN:
                try:
                    oi_raw = data[OPEN_INTEREST_COLUMN].replace(',', '').strip()
                    ivr_raw = data[IV_RANK_COLUMN].replace('%', '').strip()
                    iv30_raw = data[IV_30D_COLUMN].replace('%', '').strip()

                    open_interest_value = int(oi_raw) if oi_raw else 0
                    iv_rank_percent = float(ivr_raw) if ivr_raw else 0.0
                    iv_30d_percent = float(iv30_raw) if iv30_raw else 0.0

                    if open_interest_value > OPEN_INTEREST_LIMIT and iv_rank_percent > IV_RANK_LIMIT:
                        print(f"Found: {data[COMPANY_SYMBOL_COLUMN]} | IV Rank: {iv_rank_percent}% | OI: {open_interest_value}")
                        
                        result_item: IVItem = {
                            "company_name": data[COMPANY_NAME_COLUMN],
                            "company_symbol": data[COMPANY_SYMBOL_COLUMN],
                            "iv_rank": iv_rank_percent,
                            "iv_30d": iv_30d_percent,
                            "open_interest": open_interest_value
                        }
                        high_iv_items.append(result_item)
                except (ValueError, IndexError) as e:
                    print(f"Excountered unexoected item in IV rankings, skipping. {e}")
                    continue 
                    
        await browser.close()
    return high_iv_items
    
async def main(args):
    telegram_bot = Bot(token=args.telegram_api_token)
    high_iv_items = await parseIVDataFrom(IV_STATS_URL)
    
    if not high_iv_items:
        print("No tickers matched the criteria.")
        return

    for item in high_iv_items:
        message = (
            f"🚀 **High IV Rank Alert**\n"
            f"Stock: {item['company_name']} ({item['company_symbol']})\n"
            f"IV Rank: {item['iv_rank']}%\n"
            f"IV (30d): {item['iv_30d']}%\n"
            f"Open Interest: {item['open_interest']:,}"
        )
        await postTelegramNotification(message, telegram_bot, args.telegram_notification_group_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("IV_Rank_monitor")
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to execute IV monitor with error={e}")
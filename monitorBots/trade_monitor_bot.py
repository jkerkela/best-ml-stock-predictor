import argparse
import asyncio
import hashlib
import json
from urllib.parse import urlparse

from google import genai
from playwright.async_api import async_playwright

from telegram import Bot

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot_common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk

JSON_TRADE_DATE_KEY = "DATE"
JSON_COMPANY_SYMBOL = "SYMBOL"
JSON_TRADE_SIZE= "SIZE"
JSON_TRADE_TYPE = "TYPE"
JSON_TRADE_FOUND = "TRADE_FOUND"

REQUEST_TIMEOUT_SECONDS = 30

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

# Some trade sources (e.g. capitoltrades.com) reject non-browser user agents
BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

# The API guarantees the response conforms to this schema
TRADE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        JSON_TRADE_FOUND: {"type": "BOOLEAN"},
        JSON_COMPANY_SYMBOL: {"type": "STRING"},
        JSON_TRADE_TYPE: {"type": "STRING", "enum": ["BUY", "SELL"]},
        JSON_TRADE_SIZE: {"type": "STRING"},
        JSON_TRADE_DATE_KEY: {"type": "STRING"},
    },
    "required": [JSON_TRADE_FOUND],
}

PROMPT = """
Find the topmost (most recent) trade from the DOCUMENT below.
Report the traded company symbol, trade type, trade size and trade date.
For the date, use the trade execution date (the date the trade was made),
not the filing or publication date.
Only report a trade that is explicitly listed in the document with its type and size.
Do not infer or guess trades from news headlines or summary statistics.
If the document contains no explicitly listed trades, set TRADE_FOUND to false.

DOCUMENT:
{context}
"""


async def loadDataSourceFromWeb(web_page):
    # Rendered with a browser because sources like capitoltrades.com build the trade table with JavaScript
    print("Loading latest trade info")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(user_agent=BROWSER_USER_AGENT)
            await page.goto(web_page, wait_until="load", timeout=REQUEST_TIMEOUT_SECONDS * 1000)
            # Give client-side rendering a moment to fill in dynamic content
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            # Best effort wait for a data table to render (not all sources have one)
            try:
                await page.wait_for_selector("tbody tr", timeout=20000)
            except Exception:
                print("No table rows appeared, continuing with the content that rendered")
            # Scroll down to trigger lazy-loaded content, until the page text settles
            text = await page.locator("body").inner_text()
            stable_reads = 0
            for _ in range(12):
                await page.mouse.wheel(0, 4000)
                await page.wait_for_timeout(2500)
                new_text = await page.locator("body").inner_text()
                if len(new_text) == len(text):
                    stable_reads += 1
                    if stable_reads >= 2:
                        break
                else:
                    stable_reads = 0
                text = new_text
        except Exception as e:
            print(f"Failed to retrieve trade page with error: {e}")
            return None
        finally:
            await browser.close()
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def normalizeTrade(trade):
    # Normalize LLM output formatting so the same trade always compares equal
    for key in (JSON_COMPANY_SYMBOL, JSON_TRADE_TYPE):
        if isinstance(trade.get(key), str):
            trade[key] = trade[key].strip().upper()
    if isinstance(trade.get(JSON_TRADE_SIZE), str):
        trade[JSON_TRADE_SIZE] = trade[JSON_TRADE_SIZE].replace(",", "").replace(" ", "")
    return trade

def queryLatestTradeAsJson(gemini_client, page_text):
    """Returns (query_succeeded, trade). trade is None when the page holds no trade info."""
    print("Querying latest stock trade from LLM")
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=PROMPT.format(context=page_text),
            config={
                "response_mime_type": "application/json",
                "response_schema": TRADE_RESPONSE_SCHEMA,
            },
        )
        parsed_json = json.loads(response.text)
    except Exception as e:
        print(f"Failed to query trade from Gemini with error: {e}")
        return False, None
    if not parsed_json.get(JSON_TRADE_FOUND):
        print("No trade information found in the page content")
        return True, None
    # Guard against hallucination: the reported symbol must literally appear in the page
    symbol = parsed_json.get(JSON_COMPANY_SYMBOL, "")
    if not symbol or symbol.strip().upper() not in page_text.upper():
        print(f"Reported symbol '{symbol}' not present in the page content, discarding as hallucination")
        return False, None
    parsed_json.pop(JSON_TRADE_FOUND, None)
    return True, normalizeTrade(parsed_json)

async def main(args):
    print("Running stock trades monitor")
    gemini_client = genai.Client(api_key=args.google_api_key)
    telegram_bot = Bot(token=args.telegram_api_token)

    parsed_url = urlparse(args.source_url_for_trades)
    last_url_segment = parsed_url.path.rstrip('/').split('/')[-1]
    last_page_hash = None

    while True:
        page_text = await loadDataSourceFromWeb(args.source_url_for_trades)
        if page_text is None:
            pass
        elif hashlib.sha256(page_text.encode()).hexdigest() == last_page_hash:
            print("Page content unchanged, skipping LLM query")
        else:
            query_succeeded, latest_trade = queryLatestTradeAsJson(gemini_client, page_text)
            if query_succeeded:
                # Only skip future queries for this content once it was analyzed successfully
                last_page_hash = hashlib.sha256(page_text.encode()).hexdigest()
            if latest_trade is not None:
                latest_saved_trade = None
                try:
                    latest_saved_trade = loadObjectFromDisk(last_url_segment)
                except FileNotFoundError:
                    pass
                if latest_trade != latest_saved_trade:
                    complete_message = f"Found new trade from: {args.source_url_for_trades} as:\n {str(latest_trade)}"
                    if await postTelegramNotification(complete_message, telegram_bot, args.telegram_notification_group_id):
                        saveObjectToDisk(latest_trade, last_url_segment)
                else:
                    print("Latest trade already posted, skipping posting")
        if args.single_run:
            break
        await asyncio.sleep(600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("trading_activity_monitor")
    parser.add_argument('--google_api_key', required=True)
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    parser.add_argument('--source_url_for_trades', required=True, help='data source shall be URL that contains trade information on html GET request')
    parser.add_argument('--single_run', dest='single_run',
        help='Set to run monitoring only once, if not set the monitoring will be run in loop in 10 min intervals', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to execute trade monitor with error={e}")

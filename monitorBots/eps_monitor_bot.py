import time
from typing import TypedDict
import argparse
import asyncio

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from telegram import Bot

URL = "https://www.nasdaq.com/market-activity/earnings/daily-earnings-surprise"

EPS_DATA_PARENT_ELEMENT = "jupiter22-daily-earnings-surprise__data-container jupiter22-daily-earnings-surprise__data-container--exceeded loaded"
ACCEPT_COOKIES_BUTTON = '#onetrust-accept-btn-handler'
EPS_EXCEEDED_PARENT_ELEMENT = "daily-earnings-surprise__data"
TABLE_PARENT_ELEMENT = 'nsdq-table-sort'
SHADOW_ROOT_INSIDE_TABLE_PARENT_TRIGGER = "return arguments[0].shadowRoot.innerHTML"
EPS_ITEM_ELEMENT = "table-row"
EPS_ITEM_COMPANY_DETAIL_ELEMENT = "table-cell fixed-column-size- text-align-left"
EPS_ITEM_EPS_DETAIL_ELEMENT = "table-cell fixed-column-size- text-align-left"

EPS_ITEMS_TO_FETCH = 5
EPS_SURPRISE_THRESHOLD = 30

class EPSItem(TypedDict):
    company_name: str
    company_symbol: str
    eps_surprise_percent: float

def parseEPSDataFrom(web_page):
    print("Loading latest trade info")
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive"
    }})
    results = []
    try:
        driver.get(web_page)
        wait = WebDriverWait(driver, 20)
        cookiesButton = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ACCEPT_COOKIES_BUTTON)))
        cookiesButton.click()
        
        eps_exceeded_element = driver.find_element(By.CLASS_NAME, EPS_EXCEEDED_PARENT_ELEMENT)
        eps_table_parent = eps_exceeded_element.find_element(By.TAG_NAME, TABLE_PARENT_ELEMENT)
        shadow_root_html = driver.execute_script(SHADOW_ROOT_INSIDE_TABLE_PARENT_TRIGGER, eps_table_parent)
        driver.quit()
        soup = BeautifulSoup(shadow_root_html, "html.parser")
        EPS_entries = soup.find_all("div", class_=EPS_ITEM_ELEMENT)[:5]
        for elem in EPS_entries:
            company_details_elems = elem.find_all("div", class_=EPS_ITEM_COMPANY_DETAIL_ELEMENT)
            eps_surprise_percent_elem = elem.find_all("div", class_=EPS_ITEM_EPS_DETAIL_ELEMENT)[-1]
            eps_surprise = float(eps_surprise_percent_elem.text.replace("\n", "").strip()
            if eps_surprise > EPS_SURPRISE_THRESHOLD:
                result: EPSItem = {
                    "company_name": company_details_elems[1].text.replace("\n", ""),
                    "company_symbol": company_details_elems[0].text.replace("\n", ""),
                    "eps_surprise_percent": eps_surprise)
                }
                results.append(result)
    except Exception as e:
        print(f"Failed to fetch EPS data from source with exception: {e}")
    return results
    
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
                await asyncio.sleep(5)
            else:
                print(f"DEBUG: message send attempt failed with error: {e}, retry attemps exceeded")
                return False
                
async def main(args):
    telegram_bot = Bot(token=args.telegram_api_token)
    eps_items_list = parseEPSDataFrom(URL)
    for item in eps_items_list:
        message = f"Found company: {item["company_name"]} ({item["company_symbol"]}) with EPS surprise of {item["eps_surprise_percent"]}%"
        await postNotification(message, telegram_bot, args.telegram_notification_group_id)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("EPS_monitor")
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass

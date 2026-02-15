import os
import time
import argparse
import asyncio
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import json
import re
import pickle

from transformers import pipeline
from huggingface_hub import login
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from telegram import Bot

from huggingface_hub import snapshot_download

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot_common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk

JSON_TRADE_DATE_KEY = "DATE"
JSON_COMPANY_SYMBOL = "SYMBOL"
JSON_TRADE_SIZE= "SIZE"
JSON_TRADE_TYPE = "TYPE"

DISK_FILE_PREFIX_NAME_OF_LATEST_TRADE = "latest_trade"

PROMPT = """
INSTRUCTIONS: \n
Find UPMOST trade from the DOCUMENT below. \n
If you don't find trade information from DOCUMENT, return EMPTY STRING. \n
REPLY ONLY WITH UPMOST TRADE AS JSON:\n
{{\n
    "SYMBOL": "<company>",\n
    "TYPE": "<one of 'BUY', 'SELL'>",\n
    "SIZE": "<trade size>",\n
    "DATE": "<date>",\n
}}\n

DOCUMENT: \n
{context}\n

json:
"""

        
def loadDataSourceFromWeb(web_page):
    print("Loading latest trade info")
    response = requests.get(web_page)
    html_content = response.content
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return Document(text)
    
def queryLatestTradeAsJson(llm, data):
    print("Querying latest stock trade from LLM")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    split_docs = splitter.split_documents([data])
    prompt = PromptTemplate.from_template(PROMPT)
    for doc in split_docs:
        messages = prompt.invoke({"context": doc.page_content})
        response = llm(messages.to_string(), max_new_tokens=50, do_sample=False, top_p=None, temperature=None)
        if isinstance(response, list) and len(response) > 0:
            match = re.search(r'(\{.*?\})', response[0].get("generated_text", ""), re.DOTALL)
            if match:
                first_json_str = match.group(1)
                parsed_json = json.loads(first_json_str)
                return parsed_json
            else:
                print("No valid JSON found, continuing parsing")
        else:
            print("Warning: Unexpected response format from LLM.")
    return None
    
async def main(args):
    print("Running stock trades monitor")
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    llm = pipeline("text-generation",
        model=model_name, model_kwargs={"torch_dtype": "bfloat16"},
        device_map="cpu",
        return_full_text=False
    )
    telegram_bot = Bot(token=args.telegram_api_token)
    
    while True:
        source_data = loadDataSourceFromWeb(args.source_url_for_trades)
        latest_trade = queryLatestTradeAsJson(llm, source_data)
        
        latest_saved_trade = None
        parsed_url = urlparse(args.source_url_for_trades)
        last_url_segment = parsed_url.path.rstrip('/').split('/')[-1]
        try:
            latest_saved_trade = loadObjectFromDisk(last_url_segment)
        except: 
            pass
        do_post_latest_trade = True if not latest_trade == latest_saved_trade else False
        if do_post_latest_trade:
            complete_message = f"Found new trade from: {args.source_url_for_trades} as:\n {str(latest_trade)}"
            if await postTelegramNotification(complete_message, telegram_bot, args.telegram_notification_group_id):
                saveObjectToDisk(latest_trade, last_url_segment)
        else:
            print("Latest trade already posted, skipping posting")
        if args.single_run:
            break
        time.sleep(600)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("trading_activity_monitor")
    parser.add_argument('--huggingface_api_key', required=True)
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

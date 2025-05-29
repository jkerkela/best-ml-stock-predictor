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

parser = argparse.ArgumentParser("trading_activity_monitor")
parser.add_argument('--langsmith_api_key', required=True)
parser.add_argument('--huggingface_api_key', required=True)
parser.add_argument('--telegram_api_token', required=True)
parser.add_argument('--telegram_notification_group_id', required=True)
parser.add_argument('--source_url_for_trades', required=True, help='data source is appended shall be URL that contains trade information on html GET request')
args = parser.parse_args()

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


def saveTradeObjectToDisk(object_, file_prefix=DISK_FILE_PREFIX_NAME_OF_LATEST_TRADE):
    with open(file_prefix + ".pkl", "wb") as f:
        pickle.dump(object_, f)
        
def loadTradeObjectFromDisk(file_prefix=DISK_FILE_PREFIX_NAME_OF_LATEST_TRADE):    
    with open(file_prefix + ".pkl", "rb") as f:
        return pickle.load(f)
        
def loadDataSourceFromWeb(web_page):
    response = requests.get(web_page)
    html_content = response.content
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return Document(text)
    
def queryLatestTradeAsJson(llm, data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    split_docs = splitter.split_documents([data])
    prompt = PromptTemplate.from_template(PROMPT)
    for doc in split_docs:
        print(f"DEBUG: {doc}")
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
    
async def postNotification(message, telegram_bot, notification_group, source_url):
    print("Executing the async notification posting loop")
    retries = 3
    for attempt in range(retries):
        try:
            complete_message = f"Found new trade from: {source_url} as:\n {str(message)}"
            await telegram_bot.send_message(chat_id=notification_group, text=complete_message)
            print(f"Sent message to notification group as {complete_message}")
            return True
        except Exception as e:
            if attempt < retries - 1:
                print(f"DEBUG: message send attempt failed with error: {e}, retrying...")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
            else:
                print(f"DEBUG: message send attempt failed with error: {e}, retry attemps exceeded")
                return False
    
async def main():
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = args.langsmith_api_key
    login(token=args.huggingface_api_key)
    
    source_data = loadDataSourceFromWeb(args.source_url_for_trades)
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    llm = pipeline("text-generation",
        model=model_name, model_kwargs={"torch_dtype": "bfloat16"},
        device_map="cpu",
        return_full_text=False
    )
    telegram_bot = Bot(token=args.telegram_api_token)
    
    latest_trade = queryLatestTradeAsJson(llm, source_data)
    
    latest_saved_trade = None
    parsed_url = urlparse(args.source_url_for_trades)
    last_url_segment = parsed_url.path.rstrip('/').split('/')[-1]
    try:
        latest_saved_trade = loadTradeObjectFromDisk(last_url_segment)
    except: 
        pass
    do_post_latest_trade = True if not latest_trade == latest_saved_trade else False
    if do_post_latest_trade:
        if await postNotification(latest_trade, telegram_bot, args.telegram_notification_group_id, args.source_url_for_trades):
            saveTradeObjectToDisk(latest_trade, last_url_segment)

    
if __name__ == "__main__":
    try:
        while True:
            asyncio.run(main())
            time.sleep(600)
    except KeyboardInterrupt:
        pass

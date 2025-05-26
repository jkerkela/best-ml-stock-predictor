import argparse
import pickle
import json
import asyncio
import time

from enum import Enum
from typing import TypedDict, Tuple
from functools import partial

import requests
from bs4 import BeautifulSoup

from scipy.special import softmax
import torch
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import ToolMessage, HumanMessage

from langchain.tools import tool
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import login

from telegram import Bot


parser = argparse.ArgumentParser("stock_predict")
parser.add_argument('--huggingface_api_key', required=True)
parser.add_argument('--telegram_api_token', required=True)
parser.add_argument('--telegram_notification_group_id', required=True)
parser.add_argument('--tickers', nargs='+', required=True, help='The tickers to monitor')
args = parser.parse_args()

NEWS_ITEMS_TO_FETCH_PER_TICKER = 1
SCRAPING_SOURCE_BASE_URL = "https://www.stocktitan.net"
SCRAPING_SOURCE_NEWS_URL = SCRAPING_SOURCE_BASE_URL + "/news/"
NEWS_ITEM_SUMMARY_ELEMENT = "companies-card-summary"
NEWS_ITEM_URL_ELEMENT = "text-gray-dark feed-link"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RELEVANT_ARTICLE_JSON_KEY_NAME = "relevant"
NEWS_ITEM_REVEVANCE_CHECK_PROMPT= """
Check if the information in the context has potential impact to stock price of the company.
Give answer in json format as:\n
{{\n
    "{key_name}": "<one of 'YES', 'NO'",\n
}}\n

Context: {context}\n

json:
"""

def saveNewsObjectToDisk(object_, pickle_name="stock_news_items"):
    with open(pickle_name + ".pkl", "wb") as f:
        pickle.dump(object_, f)
        
def loadNewsObjectFromDisk(pickle_name="stock_news_items"):    
    with open(pickle_name + ".pkl", "rb") as f:
        return pickle.load(f)
        
class ItemStatus(Enum):
    CHECK_RELEVANCE = 0
    DO_SENTIMENT_ANALYSIS = 1
    SENTIMENT_DONE = 2
    IGNORE = 3
    POSTED = 4

class NewsItem(TypedDict):
    news_url: str
    news_summary: str
    item_status: ItemStatus
    sentiment: Tuple[str, float]
    
class LLMState(TypedDict):
    latest_fetched_news_per_ticker: dict[str, [NewsItem]]
    

def getLatestNewsItems(state: LLMState, tickers):
    print("DEBUG: Getting latest news items")
    try:
        state["latest_fetched_news_per_ticker"] = loadNewsObjectFromDisk()
    except: 
        pass
    for ticker in tickers:
        print(f"DEBUG: Getting latest news items for {ticker}")
        news_dir_url = SCRAPING_SOURCE_NEWS_URL + ticker
        response = requests.get(news_dir_url)
        if response.status_code != 200:
            print("Failed to retrieve page:", response.status_code)
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        article_summaries = soup.find_all("div", class_=NEWS_ITEM_SUMMARY_ELEMENT)
        article_urls = soup.find_all("a", class_=NEWS_ITEM_URL_ELEMENT)
        latest_article_summaries = article_summaries[:NEWS_ITEMS_TO_FETCH_PER_TICKER]
        latest_article_urls = article_urls[:NEWS_ITEMS_TO_FETCH_PER_TICKER]
        for index, url in enumerate(latest_article_urls):
            full_url = SCRAPING_SOURCE_BASE_URL + url['href']
            summary_text = latest_article_summaries[index].get_text(separator=" ", strip=True)
            if ticker in state["latest_fetched_news_per_ticker"]:
                for news_item in state["latest_fetched_news_per_ticker"][ticker]:
                    if news_item["news_url"] != full_url:
                        news_item: NewsItem = {
                            "news_url" : full_url,
                            "news_summary" : summary_text, 
                            "item_status" : ItemStatus.CHECK_RELEVANCE, 
                            "sentiment" : ("undefined", 0)
                        }
                        state["latest_fetched_news_per_ticker"][ticker].append(news_item)
            else:
                news_item: NewsItem = {
                    "news_url" : full_url,
                    "news_summary" : summary_text, 
                    "item_status" : ItemStatus.CHECK_RELEVANCE, 
                    "sentiment" : ("undefined", 0)
                }
                state["latest_fetched_news_per_ticker"][ticker] = [news_item]
    saveNewsObjectToDisk(state["latest_fetched_news_per_ticker"])

#TODO: this can be combined to postNotification agentic step
def checkNewsItemsRelevance(state: LLMState, relevance_check_llm):
    print("DEBUG: Filtering relevant news items")
    rag_prompt = PromptTemplate.from_template(NEWS_ITEM_REVEVANCE_CHECK_PROMPT)
    for ticker in state["latest_fetched_news_per_ticker"]:
        for news_item in state["latest_fetched_news_per_ticker"][ticker]:
            if news_item["item_status"] == ItemStatus.CHECK_RELEVANCE:
                print(f"DEBUG: Doing relevance check for:\n {news_item["news_summary"]}")
                messages = rag_prompt.invoke({"context": news_item["news_summary"], "key_name": RELEVANT_ARTICLE_JSON_KEY_NAME})
                response = relevance_check_llm(messages.to_string(), max_new_tokens=24, do_sample=False, top_p=None, temperature=None)
                print(f'Relevance check result: {response}')
                if isinstance(response, list) and len(response) > 0:
                    print(f"DEBUG: Check query analysis answer: {response}")
                    response_text = response[0].get("generated_text", "")
                    try:
                        response_val = json.loads(response_text)[RELEVANT_ARTICLE_JSON_KEY_NAME]
                        if response_val == "YES":
                            news_item["item_status"] = ItemStatus.DO_SENTIMENT_ANALYSIS
                        else:
                            news_item["item_status"] = ItemStatus.IGNORE
                    except json.JSONDecodeError as e:
                        print(f"Error parsing the response with error: {e}")
                else:
                    print("Warning: Unexpected response format from LLM.")
    saveNewsObjectToDisk(state["latest_fetched_news_per_ticker"])
    
def analyzeNewsItems(state: LLMState):
    print("DEBUG: Analyzing news items")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.to(DEVICE)
    for ticker in state["latest_fetched_news_per_ticker"]:
        for news_item in state["latest_fetched_news_per_ticker"][ticker]:
            if news_item["item_status"] == ItemStatus.DO_SENTIMENT_ANALYSIS:
                print(f"DEBUG: Doing sentiment analysis for:\n {news_item["news_summary"]}")
                inputs = tokenizer(news_item["news_summary"], return_tensors="pt", truncation=True, padding=True)
                inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                    scores = {
                    k: v
                    for k, v in zip(
                        model.config.id2label.values(),
                        softmax(logits.detach().cpu().numpy().squeeze()),
                    )
                }
                sentimentFinbert = max(scores, key=scores.get)
                probabilityFinbert = max(scores.values())
                print(f'Sentiment: {sentimentFinbert}, probabilities: {probabilityFinbert}')
                news_item["sentiment"] = (sentimentFinbert, probabilityFinbert)
                news_item["item_status"] = ItemStatus.SENTIMENT_DONE
    saveNewsObjectToDisk(state["latest_fetched_news_per_ticker"])
            

@tool
async def postNotification_test(message: str, telegram_api_token, notification_group: str):
    """Post message to notification group on telegram.

    Args:
        message(str): message to post
        telegram_api_token (str): bot api token
        notification_group (str): the group id to post notificatiokn to
    
    """
    telegram_bot = Bot(token=telegram_api_token)
    print("DEBUG: Executing the notification posting tool")
    await postNotification(message, telegram_bot, notification_group)

   

async def postNotification(message, telegram_bot, notification_group):
    print("DEBUG: Executing the async notification posting loop")
    retries = 3
    for attempt in range(retries):
        try:
            await telegram_bot.send_message(chat_id=notification_group, text=message)
            print("DEBUG: send a free slot message to notification group")
            return True
        except Exception as e:
            if attempt < retries - 1:
                print(f"DEBUG: message send attempt failed with error: {e}, retrying...")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
            else:
                print(f"DEBUG: message send attempt failed with error: {e}, retry attemps exceeded")
                return False
                    
async def postNewsItems(state: LLMState, agent_executor, telegram_api_token, notification_group):
    print("DEBUG: posting news items")
    for ticker in state["latest_fetched_news_per_ticker"]:
        for news_item in state["latest_fetched_news_per_ticker"][ticker]:
            if news_item["item_status"] == ItemStatus.SENTIMENT_DONE:
                print("DEBUG: Found news item with sentiment")
                result = await agent_executor.ainvoke(
                    {"messages": [HumanMessage(content=f"Post the notification with items: message='New news item for {ticker} with positive sentiment found in: {news_item["news_url"]}', telegram_api_token={telegram_api_token}, notification_group={notification_group}. ")]}
                )
                print(f"DEBUG: agent return: {result}")
                news_item["item_status"] = ItemStatus.POSTED
    saveNewsObjectToDisk(state["latest_fetched_news_per_ticker"])

async def main(): 
    login(token=args.huggingface_api_key)
    
    model_name = "meta-llama/Llama-3.2-1B"
    llm = pipeline("text-generation",
        model=model_name, model_kwargs={"torch_dtype": "bfloat16"},
        device_map="cpu",
        temperature=0.1,
        return_full_text=False
    )

    tools = [postNotification_test]
    agentic_llm = ChatOllama(
        model="llama3.2",
        temperature=0
    ).bind_tools(tools)
    agent_executor = create_react_agent(agentic_llm, tools=tools)
    #TODO: add conditionals here between the steps, then we don't need to check it from status anymore
    graph_builder = StateGraph(LLMState).add_sequence([
        ("getLatestNewsItems", partial(getLatestNewsItems, tickers=args.tickers)),
        ("checkNewsItemsRelevance", partial(checkNewsItemsRelevance, relevance_check_llm=llm)),
        ("analyzeNewsItems", partial(analyzeNewsItems)),
        ("postNewsItems", partial(postNewsItems, agent_executor=agent_executor, telegram_api_token=args.telegram_api_token, notification_group=args.telegram_notification_group_id))
    ])
    graph_builder.add_edge(START, "getLatestNewsItems")
    graph = graph_builder.compile()

    while True:
        result = await graph.ainvoke({ "latest_fetched_news_per_ticker": {}})
        print(result)
        time.sleep(60)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
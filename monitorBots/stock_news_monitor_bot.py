import argparse
import pickle
import json
import asyncio
import time
import requests
from bs4 import BeautifulSoup

from enum import Enum
from typing import TypedDict, Tuple
from functools import partial

import torch
from scipy.special import softmax
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import ToolMessage, HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import login

from telegram import Bot
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot_common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk


NEWS_ITEMS_TO_FETCH_PER_TICKER = 1
SCRAPING_SOURCE_BASE_URL = "https://www.stocktitan.net"
SCRAPING_SOURCE_NEWS_URL = SCRAPING_SOURCE_BASE_URL + "/news/"
NEWS_ITEM_SUMMARY_ELEMENT = "companies-card-summary"
NEWS_ITEM_URL_ELEMENT = "text-gray-dark feed-link"
SENTIMENT_THRESHOLD = 0.8

DISK_FILE_NAME_PREFIX = "stock_news_items"

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
        
class ItemStatus(Enum):
    CHECK_RELEVANCE = 1
    DO_SENTIMENT_ANALYSIS = 2
    SENTIMENT_DONE = 3
    POSTED = 4
    IGNORE = 5
    UNDEFINED = 6

class NewsItem(TypedDict):
    news_url: str
    news_summary: str
    item_status: ItemStatus
    sentiment: Tuple[str, float]
    
class LLMState(TypedDict):
    latest_fetched_news_per_ticker: dict[str, [NewsItem]]
    

def getLatestNewsItems(state: LLMState, tickers):
    print("Getting latest news items")
    try:
        state["latest_fetched_news_per_ticker"] = loadObjectFromDisk(DISK_FILE_NAME_PREFIX)
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
    saveObjectToDisk(state["latest_fetched_news_per_ticker"], DISK_FILE_NAME_PREFIX)
    return state

#TODO: this can be combined to postNotification agentic step
def checkNewsItemsRelevance(state: LLMState, relevance_check_llm):
    print("Filtering relevant news items")
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
    saveObjectToDisk(state["latest_fetched_news_per_ticker"], DISK_FILE_NAME_PREFIX)
    return state

def analyzeNewsItems(state: LLMState):
    print("Analyzing news items")
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
    return state
            

@tool
async def postNotification_test(message: str, telegram_api_token, notification_group: str):
    """Post message to notification group on telegram.

    Args:
        message(str): message to post
        telegram_api_token (str): bot api token
        notification_group (str): the group id to post notificatiokn to
    
    """
    telegram_bot = Bot(token=telegram_api_token)
    print("Executing the notification posting tool")
    await postTelegramNotification(message, telegram_bot, notification_group)

                  
def isSentimentOverThreshold(sentiment, probability):
    return (sentiment == "positive" or sentiment == "negative") and probability > SENTIMENT_THRESHOLD
      
async def postNewsItems(state: LLMState, agent_executor, telegram_api_token, notification_group):
    print("Posting news items")
    for ticker in state["latest_fetched_news_per_ticker"]:
        for news_item in state["latest_fetched_news_per_ticker"][ticker]:
            if news_item["item_status"] == ItemStatus.SENTIMENT_DONE:
                if isSentimentOverThreshold(news_item["sentiment"][0], news_item["sentiment"][1]):
                    print(f"Found news item with sentiment: {news_item["sentiment"][0]} and probability: {news_item["sentiment"][1]}")
                    result = await agent_executor.ainvoke(
                        {"messages": [HumanMessage(content=f"Post the notification with items: message='New news item for {ticker} with {news_item["sentiment"][0]} sentiment found in: {news_item["news_url"]}', telegram_api_token={telegram_api_token}, notification_group={notification_group}. ")]}
                    )
                    print(f"Agent return: {result}")
                    news_item["item_status"] = ItemStatus.POSTED
                else:
                    news_item["item_status"] = ItemStatus.IGNORE
    saveObjectToDisk(state["latest_fetched_news_per_ticker"], DISK_FILE_NAME_PREFIX)
    return state

def getNextStage(state: LLMState):
    lowest_state_found = ItemStatus.UNDEFINED
    for ticker in state["latest_fetched_news_per_ticker"]:
        for news_item in state["latest_fetched_news_per_ticker"][ticker]:
            if news_item["item_status"].value < lowest_state_found.value:
                lowest_state_found = news_item["item_status"]
    if lowest_state_found == ItemStatus.CHECK_RELEVANCE:
        return {"next_stage": "checkNewsItemsRelevance"}
    elif lowest_state_found == ItemStatus.DO_SENTIMENT_ANALYSIS:
        return {"next_stage": "analyzeNewsItems"}
    elif lowest_state_found == ItemStatus.SENTIMENT_DONE:
        return {"next_stage": "postNewsItems"}
    elif lowest_state_found == ItemStatus.POSTED:
        print("No new news items found to post")
        return {"next_stage": END}
    elif lowest_state_found == ItemStatus.IGNORE:
        print("No new news items found to post")
        return {"next_stage": END}
    else:
        return {"next_stage": "getLatestNewsItems"}

        
async def main(args):
    print("Running stock news monitor")
    login(token=args.huggingface_api_key)
    
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
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
    workflow = StateGraph(LLMState)
    workflow.add_node("getNextStage", getNextStage)
    workflow.add_node("getLatestNewsItems", partial(getLatestNewsItems, tickers=args.tickers))
    workflow.add_node("checkNewsItemsRelevance", partial(checkNewsItemsRelevance, relevance_check_llm=llm))
    workflow.add_node("analyzeNewsItems", partial(analyzeNewsItems))
    workflow.add_node("postNewsItems", partial(
        postNewsItems, 
        agent_executor=agent_executor,
        telegram_api_token=args.telegram_api_token,
        notification_group=args.telegram_notification_group_id)
    )
    
    workflow.add_edge(START, "getNextStage")
    workflow.add_edge("getLatestNewsItems", "getNextStage")
    workflow.add_edge("checkNewsItemsRelevance", "getNextStage")
    workflow.add_edge("analyzeNewsItems", "getNextStage")
    workflow.add_edge("postNewsItems", "getNextStage")
    workflow.add_conditional_edges("getNextStage", lambda state: state["next_stage"], 
    ["getLatestNewsItems", "checkNewsItemsRelevance", "analyzeNewsItems", "postNewsItems", END])
    graph = workflow.compile()

    while True:
        result = await graph.ainvoke({ "latest_fetched_news_per_ticker": {}})
        print(result)
        if args.single_run:
            break
        time.sleep(600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("stock_news_monitor")
    parser.add_argument('--huggingface_api_key', required=True)
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    parser.add_argument('--single_run', dest='single_run',
        help='Set to run monitoring only once, if not set the monitoring will be run in loop in 10 min intervals', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tickers', nargs='+', required=True, help='The tickers to monitor')
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
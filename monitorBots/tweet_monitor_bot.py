import sys
import os
import argparse
import asyncio
import requests

from telegram import Bot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk

TWEET_MONITOR_DISK_FILE_NAME = "TWEET_MONITOR_LAST_TWEET"
async def main(args):
    
    print("Running tweets monitor")
    telegram_bot = Bot(token=args.telegram_api_token)
    headers = {
        "Authorization": f"Bearer {args.twitter_API_token}"
    }
    user_url = f"https://api.twitter.com/2/users/by/username/{args.twitter_user_handle}"
    user_response = requests.get(user_url, headers=headers)

    if user_response.status_code != 200:
        print("Error getting user ID:", user_response.status_code, user_response.text)
        exit()

    user_id = user_response.json()["data"]["id"]
    tweet_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results": 5,
        "tweet.fields": "created_at,text"
    }

    response = requests.get(tweet_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        tweet = data.get("data", [{}])[0]
        tweet_content = tweet.get("text", "No tweet found")
        previously_stored_tweet = None
        try:
            previously_stored_tweet = loadObjectFromDisk(tweet_content)
        except: 
            pass
        do_post_latest_tweet = True if not tweet_content == previously_stored_tweet else False
        if do_post_latest_tweet:
            if await postTelegramNotification(tweet_content, telegram_bot, args.telegram_notification_group_id):
                saveObjectToDisk(tweet_content, TWEET_MONITOR_DISK_FILE_NAME)
            
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("tweet_monitor")
    parser.add_argument('--twitter_API_token', required=True)
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    parser.add_argument('--twitter_user_handle', required=True, help='The twitter handle to poll')
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
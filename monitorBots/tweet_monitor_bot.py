import sys
import os
import argparse
import asyncio
import requests

from telegram import Bot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot_common_tools import postTelegramNotification, saveObjectToDisk, loadObjectFromDisk

TWEET_MONITOR_STATE_DISK_FILE = "TWEET_MONITOR_STATE"
REQUEST_TIMEOUT_SECONDS = 30

def loadState():
    try:
        return loadObjectFromDisk(TWEET_MONITOR_STATE_DISK_FILE)
    except FileNotFoundError:
        return {}

def getUserId(handle, headers, state):
    user_ids = state.get("user_ids", {})
    if handle in user_ids:
        return user_ids[handle]
    user_url = f"https://api.twitter.com/2/users/by/username/{handle}"
    user_response = requests.get(user_url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    if user_response.status_code != 200:
        print("Error getting user ID:", user_response.status_code, user_response.text)
        return None
    user_id = user_response.json()["data"]["id"]
    user_ids[handle] = user_id
    state["user_ids"] = user_ids
    saveObjectToDisk(state, TWEET_MONITOR_STATE_DISK_FILE)
    return user_id

async def main(args):

    print("Running tweets monitor")
    telegram_bot = Bot(token=args.telegram_api_token)
    headers = {
        "Authorization": f"Bearer {args.twitter_API_token}"
    }
    state = loadState()
    user_id = getUserId(args.twitter_user_handle, headers, state)
    if user_id is None:
        return

    tweet_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results": 5,
        "tweet.fields": "created_at,text"
    }
    last_seen_ids = state.get("last_seen_tweet_ids", {})
    last_seen_id = last_seen_ids.get(args.twitter_user_handle)
    if last_seen_id:
        params["since_id"] = last_seen_id

    response = requests.get(tweet_url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return

    tweets = response.json().get("data", [])
    if not tweets:
        print("No new tweets")
        return

    if last_seen_id is None:
        # First run for this handle: only notify the newest tweet, not the recent history
        tweets = tweets[:1]

    # The API returns newest first, notify oldest first so the group reads chronologically
    for tweet in reversed(tweets):
        message = f"New tweet from @{args.twitter_user_handle}:\n{tweet['text']}"
        if await postTelegramNotification(message, telegram_bot, args.telegram_notification_group_id):
            last_seen_ids[args.twitter_user_handle] = tweet["id"]
            state["last_seen_tweet_ids"] = last_seen_ids
            saveObjectToDisk(state, TWEET_MONITOR_STATE_DISK_FILE)
        else:
            # Stop so the failed tweet is retried next run and chronological order is kept
            break

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
    except Exception as e:
        print(f"Failed to execute tweet monitor with error={e}")

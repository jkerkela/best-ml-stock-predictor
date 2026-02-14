import asyncio
import argparse
from datetime import datetime, time as dt_time
import time
import pytz

import stock_movers_monitor_bot
import tweet_monitor_bot
import eps_monitor_bot
import IV_monitor_bot

STOCK_MOVERS_POST_INTERVAL_IN_SECONDS = 900
STOCK_NEWS_POST_INTERVAL_IN_SECONDS = 900
STOCK_TRADE_MONITOR_POST_INTERVAL_IN_SECONDS = 3600
EPS_CHECK_INTERVAL_IN_SECONDS = 1800
IV_CHECK_INTERVALI_IN_SECONDS = 900

parser = argparse.ArgumentParser("stock_monitor")
parser.add_argument('--telegram_api_token', required=True)
parser.add_argument('--telegram_notification_group_id', required=True)
parser.add_argument('--twitter_API_token', required=True)
parser.add_argument('--twitter_user_handle', required=True, help='The twitter handle to poll')
parser.add_argument('--fmp_api_key', required=True)
parser.add_argument("--mode", choices=["premarket", "market", "postmarket"], required=False)
parser.add_argument('--single_run', dest='single_run', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

def is_weekday(d = datetime.today()):
  return d.weekday() <= 4

def isNowInTimePeriod(startTime, endTime, nowTime): 
    if startTime < endTime: 
        return nowTime >= startTime and nowTime <= endTime 
    else: 
        return nowTime >= startTime or nowTime <= endTime

def shouldPost(time_on_last_post, interval):
    return time_on_last_post == -1 or (time.time() - time_on_last_post) > interval
    
        
async def main():
    time_on_last_stock_movers_post = -1
    time_on_last_premarket_movers_post = -1
    time_on_last_trade_monitor_post = time.time() # Do not check on first run
    time_on_last_EPS_monitor_post = -1
    time_on_last_IV_monitor_post = -1
    while True:
        print("Checking if monitoring bots need to run")
        if is_weekday():
            tz = pytz.timezone("Europe/Helsinki") 
            current_time = datetime.now(tz).time()
            if shouldPost(time_on_last_premarket_movers_post, STOCK_MOVERS_POST_INTERVAL_IN_SECONDS) and isNowInTimePeriod(dt_time(14, 30, 0), dt_time(16, 30, 0), current_time):
                args.mode = "premarket"
                await stock_movers_monitor_bot.main(args)
                time_on_last_premarket_movers_post = time.time()
            elif shouldPost(time_on_last_stock_movers_post, STOCK_MOVERS_POST_INTERVAL_IN_SECONDS) and isNowInTimePeriod(dt_time(16, 30, 0), dt_time(17, 30, 0), current_time):
                args.mode = "market"
                await stock_movers_monitor_bot.main(args)
                time_on_last_stock_movers_post = time.time()
            elif shouldPost(time_on_last_stock_movers_post, STOCK_MOVERS_POST_INTERVAL_IN_SECONDS) and isNowInTimePeriod(dt_time(23, 0, 0), dt_time(0, 0, 0), current_time):
                args.mode = "postmarket"
                await stock_movers_monitor_bot.main(args)
                time_on_last_stock_movers_post = time.time()
            if shouldPost(time_on_last_EPS_monitor_post, EPS_CHECK_INTERVAL_IN_SECONDS) and (isNowInTimePeriod(dt_time(13, 30, 0), dt_time(15, 30, 0), current_time) or isNowInTimePeriod(dt_time(23, 0, 0), dt_time(1, 0, 0), current_time)):
                await eps_monitor_bot.main(args)
                time_on_last_EPS_monitor_post = time.time()
            if shouldPost(time_on_last_IV_monitor_post, IV_CHECK_INTERVALI_IN_SECONDS) and (isNowInTimePeriod(dt_time(16, 30, 0), dt_time(18, 30, 0), current_time)):
                await IV_monitor_bot.main(args)
                IV_monitor_bot = time.time()
        if shouldPost(time_on_last_trade_monitor_post, STOCK_TRADE_MONITOR_POST_INTERVAL_IN_SECONDS):
            await tweet_monitor_bot.main(args)
            time_on_last_trade_monitor_post = time.time()
        time.sleep(60)
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
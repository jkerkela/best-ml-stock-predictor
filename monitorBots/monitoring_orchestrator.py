import asyncio
import argparse
from datetime import datetime, time as dt_time
import time
import pytz

import stock_movers_monitor_bot
import eps_monitor_bot
import IV_monitor_bot
import trade_monitor_bot

STOCK_MOVERS_POST_INTERVAL_IN_SECONDS = 3600
EPS_CHECK_INTERVAL_IN_SECONDS = 1800
IV_CHECK_INTERVAL_IN_SECONDS = 3600
TRADE_MONITOR_CHECK_INTERVAL_IN_SECONDS = 3600

EASTERN_TZ = pytz.timezone('US/Eastern')

parser = argparse.ArgumentParser("stock_monitor")
parser.add_argument('--telegram_api_token', required=True)
parser.add_argument('--telegram_notification_group_id', required=True)
parser.add_argument('--fmp_api_key', required=True)
parser.add_argument('--google_api_key', required=True)
parser.add_argument('--source_url_for_trades', required=True, help='data source shall be URL that contains trade information on html GET request')
parser.add_argument("--mode", choices=["premarket", "market", "postmarket"], required=False)
parser.add_argument('--single_run', dest='single_run', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

def is_weekday():
    return datetime.now(EASTERN_TZ).weekday() <= 4

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
    time_on_last_EPS_monitor_post = -1
    time_on_last_IV_monitor_post = -1
    time_on_last_trade_monitor_post = -1
    while True:
        try:
            current_time = datetime.now(EASTERN_TZ).time()
            if is_weekday():
                if shouldPost(time_on_last_premarket_movers_post, STOCK_MOVERS_POST_INTERVAL_IN_SECONDS) and isNowInTimePeriod(dt_time(4, 00, 0), dt_time(9, 30, 0), current_time):
                    print(f"Running PRE market bot, current time in target timezone={current_time}")
                    args.mode = "premarket"
                    await stock_movers_monitor_bot.main(args)
                    time_on_last_premarket_movers_post = time.time()
                elif shouldPost(time_on_last_stock_movers_post, STOCK_MOVERS_POST_INTERVAL_IN_SECONDS) and isNowInTimePeriod(dt_time(9, 30, 0), dt_time(16, 0, 0), current_time):
                    print(f"Running live market bot, current time in target timezone={current_time}")
                    args.mode = "market"
                    await stock_movers_monitor_bot.main(args)
                    time_on_last_stock_movers_post = time.time()
                elif shouldPost(time_on_last_stock_movers_post, STOCK_MOVERS_POST_INTERVAL_IN_SECONDS) and isNowInTimePeriod(dt_time(16, 0, 0), dt_time(20, 0, 0), current_time):
                    print(f"Running POST market bot, current time in target timezone={current_time}")
                    args.mode = "postmarket"
                    await stock_movers_monitor_bot.main(args)
                    time_on_last_stock_movers_post = time.time()
                if shouldPost(time_on_last_EPS_monitor_post, EPS_CHECK_INTERVAL_IN_SECONDS) and (isNowInTimePeriod(dt_time(4, 0, 0), dt_time(9, 30, 0), current_time) or isNowInTimePeriod(dt_time(16, 0, 0), dt_time(20, 0, 0), current_time)):
                    print(f"Running EPS bot, current time in target timezone={current_time}")
                    await eps_monitor_bot.main(args)
                    time_on_last_EPS_monitor_post = time.time()
                if shouldPost(time_on_last_IV_monitor_post, IV_CHECK_INTERVAL_IN_SECONDS) and (isNowInTimePeriod(dt_time(9, 30, 0), dt_time(11, 30, 0), current_time) or isNowInTimePeriod(dt_time(14, 00, 0), dt_time(16, 0, 0), current_time)):
                    print(f"Running IV bot, current time in target timezone={current_time}")
                    await IV_monitor_bot.main(args)
                    time_on_last_IV_monitor_post = time.time()
                if shouldPost(time_on_last_trade_monitor_post, TRADE_MONITOR_CHECK_INTERVAL_IN_SECONDS) and isNowInTimePeriod(dt_time(9, 0, 0), dt_time(12, 0, 0), current_time):
                    print(f"Running trade bot, current time in target timezone={current_time}")
                    await trade_monitor_bot.main(args)
                    time_on_last_trade_monitor_post = time.time()
        except Exception as e:
            print(f"Failed to execute monitoring with error={e}")
        await asyncio.sleep(60)
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
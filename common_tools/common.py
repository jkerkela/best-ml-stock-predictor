import pickle
import asyncio
from telegram import Bot

async def postTelegramNotification(message, telegram_bot, notification_group):
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
                
def saveObjectToDisk(object_, file_prefix):
    with open(file_prefix + ".pkl", "wb") as f:
        pickle.dump(object_, f)
        
def loadObjectFromDisk(file_prefix):    
    with open(file_prefix + ".pkl", "rb") as f:
        return pickle.load(f)
import argparse
import asyncio

import imaplib
import email

from telegram import Bot

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_tools import postTelegramNotification
    
async def main(args):
    mail = imaplib.IMAP4_SSL("mail.mailo.com")
    mail.login(args.email_address, args.email_password)
    mail.select("inbox")

    result, data = mail.search(None, "UNSEEN")
    email_ids = data[0].split()
    
    telegram_bot = Bot(token=args.telegram_api_token)
    for e_id in email_ids:
        result, msg_data = mail.fetch(e_id, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        print(f"New email from: {msg['From']} - {msg['Subject']}")

        await postTelegramNotification(f"New indicator notification: {msg['Subject']}", telegram_bot, args.telegram_notification_group_id)

    mail.logout()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("mail_indicator_notification")
    parser.add_argument('--email_address', required=True)
    parser.add_argument('--email_password', required=True)
    parser.add_argument('--telegram_api_token', required=True)
    parser.add_argument('--telegram_notification_group_id', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
import datetime
import pandas
from datetime import date
from iexfinance import get_historical_data
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-sq", "--stockquote", dest="ticker",
                    help="Stock quote data to load")
args = parser.parse_args()

ticker = args.ticker
start_date = datetime.datetime.now() - datetime.timedelta(days=1*365)
end_date = str(date.today())

dataFrame = get_historical_data(ticker, start=start_date, end=end_date, output_format='pandas')
dataFrame.to_csv(ticker + "_stock_data.csv", index_label="date")
import datetime
import pandas
from datetime import date
from iexfinance import get_historical_data
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-q", "--quote", dest="stock_quote",
                    help="Stock quote data to load")
args = parser.parse_args()

stock_quote = args.stock_quote
start_date = datetime.datetime.now() - datetime.timedelta(days=1*455)
end_date = str(date.today())

dataframe = get_historical_data(stock_quote, start=start_date, end=end_date, output_format='pandas')
data_directory = str("./data/")
os.makedirs(data_directory, exist_ok=True)
dataframe.to_csv(data_directory + stock_quote + "_stock_data.csv", index_label="date")
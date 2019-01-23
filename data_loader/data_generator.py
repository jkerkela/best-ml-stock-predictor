import matplotlib.pyplot as plt
import numpy as np
import talib
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
RSI14 = talib.RSI(dataFrame["close"], 14)
RSI50 = talib.RSI(dataFrame["close"], 50)

rawDataFrame = {
	"RSI14": RSI14,
	"RSI50": RSI50,
	"stock_value": dataFrame["close"]
}
columnOrder = ["RSI14", "RSI50", "stock_value"]

collectDF = pandas.DataFrame(
            data=rawDataFrame,
            columns=columnOrder)
processedDF = collectDF.iloc[50:]
trainingDF = processedDF.iloc[150:]
validationDF = processedDF.iloc[50:]

trainingDF.to_csv("train_" + ticker + ".csv", index_label="date")
validationDF.to_csv("validation_" + ticker + ".csv", index_label="date")
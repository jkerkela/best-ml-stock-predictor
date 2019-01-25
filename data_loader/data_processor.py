import talib
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", "--source", dest="source",
                    help="source file to process")
args = parser.parse_args()


dataFrame = pd.read_csv(args.source, sep=",")

# calculate RSI values 
dataFrame["RSI14"] = talib.RSI(dataFrame["close"], 14)
dataFrame["RSI50"] = talib.RSI(dataFrame["close"], 50)

# calculate 1 week on month future values 
dataFrameIndexesWithFuture = len(dataFrame.index) - 20
for index, row in dataFrame.iterrows():
	if (index < dataFrameIndexesWithFuture):
		lookWeekAHeadLoc = index + 5
		dataFrame.loc[index,"1_week_future_value"] = dataFrame.loc[lookWeekAHeadLoc,"close"] 
		lookMonthAHeadLoc = index + 20
		dataFrame.loc[index,"1_month_future_value"] = dataFrame.loc[lookMonthAHeadLoc,"close"] 
	
# Drop indexes with empty values	
dataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# Rename and drop not needed columns
dataFrame.rename(columns={'close': 'stock_value'}, inplace=True)
dataFrame.drop(['date', 'open', 'high', 'low', 'volume'], axis=1, inplace=True)

# Divide to training and validation data
trainingDF = dataFrame.head(140)
validationDF = dataFrame.tail(40)

trainingDF.to_csv("train.csv", index=False)
validationDF.to_csv("validation.csv", index=False)

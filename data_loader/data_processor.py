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

# Divide to training, validation and test data
dataFrameIndexCount = len(dataFrame.index)
trainingDFBucketLastIndex = (int) (dataFrameIndexCount / 10 * 6)
validationDFBucketLastIndex = trainingDFBucketLastIndex + int (dataFrameIndexCount / 10 * 2)
trainingDFBucket = dataFrame.iloc[0:trainingDFBucketLastIndex]
validationDFBucket = dataFrame.iloc[(trainingDFBucketLastIndex + 1):validationDFBucketLastIndex]
testDFBucket = dataFrame.iloc[(validationDFBucketLastIndex + 1):(dataFrameIndexCount - 1)]

# this needs to be done before dropna call
predictSourceBucket = dataFrame.head(20)

dataDirectory = str("./data/")
trainingDFBucket.to_csv(dataDirectory + "train.csv", index=False)
validationDFBucket.to_csv(dataDirectory + "validation.csv", index=False)
testDFBucket.to_csv(dataDirectory + "evaluate.csv", index=False)
predictSourceBucket.to_csv(dataDirectory + "predict_source.csv", index=False)

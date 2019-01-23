import talib
import pandas
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", "--source", dest="source",
                    help="source file to process")
args = parser.parse_args()

dataFrame = pandas.read_csv(args.source, sep=",")
RSI14 = talib.RSI(dataFrame["close"], 14)
RSI50 = talib.RSI(dataFrame["close"], 50)

#put 1 week and 1 month future value delta to dataframe
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
trainingDF = processedDF.head(150)
validationDF = processedDF.tail(50)

trainingDF.to_csv("train.csv", index=False)
validationDF.to_csv("validation.csv", index=False)
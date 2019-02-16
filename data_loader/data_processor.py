import talib
import pandas as pd
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("-s", "--source", dest="source_file",
                    help="source file to process")
args = parser.parse_args()

dataframe = pd.read_csv(args.source_file, sep=",")

# Calculate RSI values 
dataframe["RSI14"] = talib.RSI(dataframe["close"], 14)
dataframe["RSI50"] = talib.RSI(dataframe["close"], 50)

# Divide last month data as predition data
predict_source_bucket = dataframe.tail(30)

# calculate 1 week on month future values 
dataframe_indexes_with_future = len(dataframe.index) - 20
for index, row in dataframe.iterrows():
	if (index < dataframe_indexes_with_future):
		month_a_head_loc = index + 20
		dataframe.loc[index,"1_month_future_value"] = dataframe.loc[month_a_head_loc,"close"] 

# Rename and drop not needed columns
dataframe.rename(columns={'close': 'stock_value'}, inplace=True)
dataframe.drop(['date', 'open', 'high', 'low', 'volume'], axis=1, inplace=True)

# Drop indexes with empty values	
dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# Divide to training, validation and test data
dataframe_index_count = len(dataframe.index)
training_DF_bucket_last_index = (int) (dataframe_index_count / 10 * 6)
validation_DF_bucket_last_index = training_DF_bucket_last_index + int (dataframe_index_count / 10 * 2)
training_DF_bucket = dataframe.iloc[0:training_DF_bucket_last_index]
validation_DF_bucket = dataframe.iloc[(training_DF_bucket_last_index + 1):validation_DF_bucket_last_index]
evaluation_DF_bucket = dataframe.iloc[(validation_DF_bucket_last_index + 1):(dataframe_index_count - 1)]

# Save data items to file
data_directory = str("./data/")
os.makedirs(data_directory, exist_ok=True)
training_DF_bucket.to_csv(data_directory + "train.csv", index=False)
validation_DF_bucket.to_csv(data_directory + "validation.csv", index=False)
evaluation_DF_bucket.to_csv(data_directory + "evaluate.csv", index=False)
predict_source_bucket.to_csv(data_directory + "predict_source.csv", index=False)

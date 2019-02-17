from IPython import display
import numpy as np
import pandas as pd
import os
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import tools.data_item_processor as dataItemProcessor
import tools.tensor_object_provider as tensorObjProvider
  
def predict_with_model(model_regressor,
                       features,
					   template_frame):
  """Predicts with model regressor.
  
  Predictions are output to result file
  
  Args:
    model_regressor: A trained regressor object (DNN type)
	features: A DataFrame that contains the features
	template_frame: A DataFrame that containing index column to be used in result file
  """

  predict_input_fn = lambda: tensorObjProvider.predict_input_fn(features)
  final_predictions = model_regressor.predict(input_fn=predict_input_fn)
  final_predictions = np.array([item['predictions'][0] for item in final_predictions])
  prediction_dataframe = pd.DataFrame(final_predictions)
  prediction_dataframe.rename(columns={prediction_dataframe.columns[0]: '1_month_future_stock_value_prediction'}, inplace=True)
  prediction_dataframe = pd.concat([template_frame, prediction_dataframe], axis=1)
  results_dir = str("./results/")
  os.makedirs(results_dir, exist_ok=True)
  prediction_dataframe.to_csv(results_dir + 'prediction.csv', index=False)
  
# Load source data  
data_directory = str("../data_loader/data/")
source_dataframe = pd.read_csv(data_directory + "predict_source.csv", sep=",")
data_features = dataItemProcessor.preprocess_features(source_dataframe)
print("Prediction data key indicators:")
display.display(data_features.describe())

# Extract dates from source data to dataframe
prediction_template_dataframe = source_dataframe["date"]

feature_columns = dataItemProcessor.construct_feature_columns(data_features)
regressor = tensorObjProvider.get_DNN_regressor(features=feature_columns)

predict_with_model(model_regressor=regressor,
                   features=data_features,
				   template_frame=prediction_template_dataframe)


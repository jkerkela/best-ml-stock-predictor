from IPython import display
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
			  
def my_input_fn(features):
  """Trains model of multiple features.
  
  Args:
    features: pandas DataFrame of features
  Returns:
    Tuple of (features, labels) for next data batch
  """
    
  # Convert pandas data into a dict of np arrays.
  features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
  # Construct a dataset, and configure batching/repeating.
  ds = Dataset.from_tensor_slices((features)) # warning: 2GB limit
  ds = ds.batch(1).repeat(1)
    
  # Return the next batch of data.
  features = ds.make_one_shot_iterator().get_next()
  return features
			  
def preprocess_features(stock_dataframe):
  """Prepares input features from California housing data set.

  Args:
    stock_dataframe: A Pandas DataFrame expected to contain data
      from the stock data set.
  Returns:
    A DataFrame that contains the features to be used for the model.
  """
  processed_features = stock_dataframe[
    ["RSI14",
     "RSI50"]]
  return processed_features

def predict_with_model(model_regressor,
                       source_data,
					   template_frame):
  """predict with model"""

  predict_input_fn = lambda: my_input_fn(source_data)
  
  final_predictions = model_regressor.predict(input_fn=predict_input_fn)
  final_predictions = np.array([item['predictions'][0] for item in final_predictions])
  predictionDataFrame = pd.DataFrame(final_predictions)
  predictionDataFrame.rename(columns={predictionDataFrame.columns[0]: '1_month_future_stock_value_prediction'}, inplace=True)
  predictionDataFrame = pd.concat([template_frame, predictionDataFrame], axis=1)
  results_dir = str("./results/")
  os.makedirs(results_dir, exist_ok=True)
  predictionDataFrame.to_csv(results_dir + 'prediction.csv', index=False)
  
def getRegressor(features,
                 learning_rate,
                 hidden_units = []):
  """get regressor model
  
  Args:
  features: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as features.
    model_type: A `String`, the model type to be trained
	learning_rate: A `float`, the learning rate.
	hidden_units: A `list` of int values, specifying the number of neurons in each layer.
   """
  
  # Create a DNNRegressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
    feature_columns=construct_feature_columns(features),
    hidden_units=hidden_units,
    optimizer=my_optimizer,
	model_dir="./models/NN_train_model"
  )
  return dnn_regressor

# Load source data  
dataDirectory = str("../data_loader/data/")
source_dataframe = pd.read_csv(dataDirectory + "predict_source.csv", sep=",")
data_features = preprocess_features(source_dataframe)
print("Prediction data key indicators:")
display.display(data_features.describe())

# Extract dates from source data to dataframe
prediction_template_dataframe = source_dataframe["date"]

regressor = getRegressor(features=data_features,
                        learning_rate=0.001,
                        hidden_units=[5,5])

predict_with_model(model_regressor=regressor,
                   source_data=data_features,
				   template_frame=prediction_template_dataframe)


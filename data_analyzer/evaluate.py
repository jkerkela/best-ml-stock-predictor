import math
from IPython import display
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
			  
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
  """Trains model of multiple features.
  
  Args:
    features: pandas DataFrame of features
    targets: pandas DataFrame of targets
    batch_size: Size of batches to be passed to the model
    shuffle: True or False. Whether to shuffle the data.
    num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
  Returns:
    Tuple of (features, labels) for next data batch
  """
    
  # Convert pandas data into a dict of np arrays.
  features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
  # Construct a dataset, and configure batching/repeating.
  ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
  ds = ds.batch(batch_size).repeat(num_epochs)
    
  # Shuffle the data, if specified.
  if shuffle:
    ds = ds.shuffle(10000)
    
  # Return the next batch of data.
  features, labels = ds.make_one_shot_iterator().get_next()
  return features, labels
			  
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

def preprocess_targets(stock_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    stock_dataframe: A Pandas DataFrame expected to contain data
      from the stock data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  processed_targets = stock_dataframe[
    ["1_month_future_value"]]
  return processed_targets
  
def evaluate_model(model_regressor,
                       evaluation_examples,
					   evaluation_targets):
  """Evaluate model"""

  evaluate_input_fn = lambda: my_input_fn(
    evaluation_examples,
    evaluation_targets["1_month_future_value"],
	num_epochs=1,
    shuffle=False)

  evaluation_predictions = model_regressor.predict(input_fn=evaluate_input_fn)
  evaluation_predictions = np.array([item['predictions'][0] for item in evaluation_predictions])
  
  root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(evaluation_predictions, evaluation_targets))

  print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
  
def getRegressor(features,
                 learning_rate,
                 hidden_units=[]):
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
	
data_directory = str("../data_loader/data/")
evaluation_data_file = pd.read_csv(data_directory + "evaluate.csv", sep=",")
evaluation_examples = preprocess_features(evaluation_data_file)
evaluation_targets = preprocess_targets(evaluation_data_file)
print("Evaluate data key indicators:")
display.display(evaluation_examples.describe())
display.display(evaluation_targets.describe())

regressor = getRegressor(features=evaluation_examples,
                        learning_rate=0.001,
                        hidden_units=[5,5])

evaluate_model(regressor,
			   evaluation_examples,
			   evaluation_targets)
import math
from argparse import ArgumentParser
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

parser = ArgumentParser()
parser.add_argument("-m", "--model_type", 
                    choices=['linear_regressor', 'neural_network'],
					default='neural_network',
					nargs='?',
                    dest="model_to_train",
                    help="Model type to train")
args = parser.parse_args()

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
  
def train_model(
    model_type,
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    hidden_units = [],
    ):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    model_type: A `String`, the model type to be trained
	learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `stock_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `stock_dataframe` to use as target for validation.
	hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  if model_type == "linear_regressor":
    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer,
		model_dir="./models/LIN_train_model_test"
    )
  elif model_type == "neural_network":
    # Create a DNNRegressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer,
		model_dir="./models/NN_train_model_test"
    )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["1_month_future_value"], 
      batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["1_month_future_value"], 
      num_epochs=1, 
      shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
      validation_examples, validation_targets["1_month_future_value"], 
      num_epochs=1, 
      shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training " + model_type + " model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    if model_type == "linear_regressor":
      regressor = linear_regressor.train(
		  input_fn=training_input_fn,
		  steps=steps_per_period,
	  )
    elif model_type == "neural_network":
      regressor = dnn_regressor.train(
          input_fn=training_input_fn,
          steps=steps_per_period
      )
	
	# Take a break and compute predictions.
    training_predictions = regressor.predict(input_fn=predict_training_input_fn)
    validation_predictions = regressor.predict(input_fn=predict_validation_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return regressor
  
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

def train_linear_model(training_examples,
                       training_targets,
                       validation_examples,
                       validation_targets):
  """Trains linear model with multiple features.
  
  Args:
  training_examples: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `stock_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `stock_dataframe` to use as target for validation.
  
  Returns:
    A `LinearRegressor` object trained on the training data.
  """
	
  return train_model(
    args.model_to_train,
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

def train_neural_network_model(training_examples,
                       training_targets,
                       validation_examples,
                       validation_targets):
  """Trains linear model with multiple features.
  
  Args:
  training_examples: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `stock_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `stock_dataframe` to use as target for validation.
  
  Returns:
    A `DNNRegressor` object trained on the training data.
  """
	
  return train_model(
    args.model_to_train,
    learning_rate=0.001,
    steps=750,
    batch_size=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets,
    hidden_units=[5,5])
  
# init environment
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load data files 
dataDirectory = str("../data_loader/data/")
trainDataFile = pd.read_csv(dataDirectory + "train.csv", sep=",")
training_examples = preprocess_features(trainDataFile)
training_targets = preprocess_targets(trainDataFile)
print("Traning data key indicators:")
display.display(training_examples.describe())
display.display(training_targets.describe())
validation_data = pd.read_csv(dataDirectory + "validation.csv", sep=",")
validation_examples = preprocess_features(validation_data)
validation_targets = preprocess_targets(validation_data)
print("Traning data key indicators:")
display.display(validation_examples.describe())
display.display(validation_targets.describe())

# Train model
if args.model_to_train == "linear_regressor":
  train_linear_model(training_examples, training_targets, validation_examples, validation_targets)

elif args.model_to_train == "neural_network":
  train_neural_network_model(training_examples, training_targets, validation_examples, validation_targets)

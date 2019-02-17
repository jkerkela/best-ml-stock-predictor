from argparse import ArgumentParser
from IPython import display
from sklearn import metrics
import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tools.data_item_processor as dataItemProcessor
import tools.tensor_object_provider as tensorObjProvider

parser = ArgumentParser()
parser.add_argument("-m", "--model_type", 
                    choices=['linear_regressor', 'neural_network'],
					default='neural_network',
					nargs='?',
                    dest="model_to_train",
                    help="Model type to train")
args = parser.parse_args()
  
def train_model(
    model_type,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    ):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    model_type: A `String`, the model type to be trained
    batch_size: A non-zero `int`, the batch size.
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

  periods = 10
  steps = 750
  steps_per_period = steps / periods
  
  feature_columns=dataItemProcessor.construct_feature_columns(training_examples)
  if model_type == "linear_regressor":
    linear_regressor = tensorObjProvider.get_linear_regressor(features=feature_columns)
  elif model_type == "neural_network":
    dnn_regressor = tensorObjProvider.get_DNN_regressor(features=feature_columns)
  
  # Create input functions.
  
  training_input_fn = lambda: tensorObjProvider.train_input_fn(
      training_examples, 
      training_targets["1_month_future_value"], 
      batch_size=10)
  predict_training_input_fn = lambda: tensorObjProvider.train_input_fn(
      training_examples, 
      training_targets["1_month_future_value"], 
      num_epochs=1, 
      shuffle=False)
  predict_validation_input_fn = lambda: tensorObjProvider.train_input_fn(
      validation_examples, validation_targets["1_month_future_value"], 
      num_epochs=1, 
      shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess loss metrics.
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
  
# init environment
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load data files 
data_directory = str("../data_loader/data/")
training_data = pd.read_csv(data_directory + "train.csv", sep=",")
training_examples = dataItemProcessor.preprocess_features(training_data)
training_targets = dataItemProcessor.preprocess_targets(training_data)
print("Traning data key indicators:")
display.display(training_examples.describe())
display.display(training_targets.describe())
validation_data = pd.read_csv(data_directory + "validation.csv", sep=",")
validation_examples = dataItemProcessor.preprocess_features(validation_data)
validation_targets = dataItemProcessor.preprocess_targets(validation_data)
print("Traning data key indicators:")
display.display(validation_examples.describe())
display.display(validation_targets.describe())

# Train model
train_model(model_type=args.model_to_train, 
            training_examples=training_examples,
		    training_targets=training_targets,
		    validation_examples=validation_examples,
		    validation_targets=validation_targets)

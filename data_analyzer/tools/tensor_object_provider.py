import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

def train_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
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
  
def predict_input_fn(features):
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
  
def get_DNN_regressor(features):
  """get regressor model
  
  Args:
  features: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as features.
   """
  
  # Create a DNNRegressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
    feature_columns=features,
    hidden_units=[5,5],
    optimizer=my_optimizer,
	model_dir="./models/NN_train_model"
  )
  return dnn_regressor
  
def get_linear_regressor(features):
  """get regressor model
  
  Args:
  features: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as features.
   """
  
  # Create a linearreRegressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.DNNRegressor(
    feature_columns=features,
    optimizer=my_optimizer,
	model_dir="./models/LIN_train_model_test"
  )
  return linear_regressor
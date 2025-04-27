import tensorflow as tf

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
  
def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
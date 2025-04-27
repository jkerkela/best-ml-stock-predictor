from IPython import display
import math
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import tools.data_item_processor as dataItemProcessor
import tools.tensor_object_provider as tensorObjProvider
  
def evaluate_model(model_regressor,
                   evaluation_examples,
				   evaluation_targets):
  """Evaluates model.
  
  In addition to evaluation, this function also prints evaluation progress information
  and a evaluation loss.
  
  Args:
    model_regressor: A trained regressor object (DNN type)
	evaluation_examples: A `DataFrame` containing one or more columns from
      `stock_dataframe` to use as input features for evaluation.
	evaluation_targets: A `DataFrame` containing exactly one column from
      `stock_dataframe` to use as target for evaluation.
  """

  evaluate_input_fn = lambda: tensorObjProvider.train_input_fn(
    evaluation_examples,
    evaluation_targets["1_month_future_value"],
	num_epochs=1,
    shuffle=False)

  evaluation_predictions = model_regressor.predict(input_fn=evaluate_input_fn)
  evaluation_predictions = np.array([item['predictions'][0] for item in evaluation_predictions])
  
  root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(evaluation_predictions, evaluation_targets))

  print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
	
data_directory = str("../data_loader/data/")
evaluation_data_file = pd.read_csv(data_directory + "evaluate.csv", sep=",")
evaluation_examples = dataItemProcessor.preprocess_features(evaluation_data_file)
evaluation_targets = dataItemProcessor.preprocess_targets(evaluation_data_file)
print("Evaluate data key indicators:")
display.display(evaluation_examples.describe())
display.display(evaluation_targets.describe())

feature_columns = dataItemProcessor.construct_feature_columns(evaluation_examples)
regressor = tensorObjProvider.get_DNN_regressor(features=feature_columns)

evaluate_model(regressor,
			   evaluation_examples,
			   evaluation_targets)
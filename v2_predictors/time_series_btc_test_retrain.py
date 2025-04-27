import math
import os

import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit.visualization import plot_predictions


#dataset definitions
timestamp_column = "date"
target_columns = ["price"]

#model definitions
TTM_MODEL_REVISION = "512-96-ft-r2.1"
context_length = 512 # the max context length for the 512-96 model
prediction_length = 96  # the max forecast length for the 512-96 model
DATA_FILE_PATH = "hf://datasets/gauss314/bitcoin_daily/data_btc.csv"

# Return this percent of the original dataset when getting train/test splits.
fewshot_fraction = 0.05

# Output directory for writing evaluation results.
OUT_DIR = "ttm_results/"
            
def run_model():
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read in the data from the downloaded file.
    input_df = pd.read_csv(
        DATA_FILE_PATH,
        parse_dates=[timestamp_column],  # Parse the timestamp values as dates.
    )
    # Fill NA/NaN values by propagating the last valid value.
    input_df = input_df.ffill()

    # Show the last few rows of the dataset.
    input_df.tail()

    for target_column in target_columns:
        input_df.plot(x=timestamp_column, y=target_column, figsize=(20, 3))


        split_config = {"train": 0.6, "test": 0.2}

        column_specifiers = {
            "timestamp_column": timestamp_column,
            "id_columns":[],
            "target_columns": target_columns,
        }

        tsp = TimeSeriesPreprocessor(
            **column_specifiers,
            context_length=context_length,
            prediction_length=prediction_length,
            scaling=True,
            encode_categorical=False,
            scaler_type="standard",
            freq="d",
        )

        train_df, valid_df, test_df = prepare_data_splits(input_df, context_length=context_length, split_config=split_config) 

        # Instantiate the model.
        set_seed(42)
        finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2",  # Name of the model on HuggingFace.
            num_input_channels=tsp.num_input_channels,
            prediction_channel_indices=tsp.prediction_channel_indices,
            fcm_use_mixer=True,
            fcm_context_length=10,
            enable_forecast_channel_mixing=True,
            decoder_mode="mix_channel",
        )

        print(
            "Number of params before freezing backbone",
            count_parameters(finetune_forecast_model),
        )

        # Freeze the backbone of the model
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(finetune_forecast_model),
        )

        train_dataset, valid_dataset, test_dataset = get_datasets(
            tsp,
            input_df,
            split_config,
            fewshot_fraction=fewshot_fraction,
            fewshot_location="first",
            use_frequency_token=finetune_forecast_model.config.resolution_prefix_tuning,
        )
        print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

        # Important parameters
        learning_rate: float = 0.002 # decrease for more accurate convergation of training e.g. 0.0004
        num_epochs: int =  4 # increase to have more accurate fitting to training data e.g. 10
        patience: int = 10
        batch_size: int = 256

        print(f"Using learning rate = {learning_rate}")
        finetune_forecast_args = TrainingArguments(
            output_dir=os.path.join(OUT_DIR, "output"),
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=4,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            use_cpu=device != "cuda",
        )

        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=patience,  # Number of epochs with no improvement after which to stop
            early_stopping_threshold=0.00001,  # Minimum improvement required to consider as improvement
        )
        tracking_callback = TrackingCallback()

        # Optimizer and scheduler
        optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=num_epochs,
            steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
        )

        finetune_forecast_trainer = Trainer(
            model=finetune_forecast_model,
            args=finetune_forecast_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
        )

        finetune_forecast_trainer.train()

        # generate forecasts using the finetuned model
        pipeline = TimeSeriesForecastingPipeline(
            finetune_forecast_model,
            device=device,  # Specify your local GPU or CPU.
            feature_extractor=tsp,
            batch_size=batch_size,
        )

        # Make a forecast on the target column given the input data.
        finetune_forecast = pipeline(test_df)
        finetune_forecast.head()

        plot_predictions(
            input_df=test_df,
            predictions_df=finetune_forecast,
            timestamp_column=timestamp_column,
            freq=tsp.freq,
            plot_dir=None,
            plot_prefix="Test",
            channel=target_column, 
            plot_context=2 * prediction_length,
            indices= [300], # increase if needed
            num_plots=4,
        )

        historical_df = input_df.iloc[-context_length - prediction_length : -prediction_length].copy()
        controls_df = input_df.iloc[-prediction_length:][column_specifiers["target_columns"]].copy()

        # Create a pipeline.
        pipeline = TimeSeriesForecastingPipeline(
            finetune_forecast_trainer.model,
            feature_extractor=tsp,
            device=device,  # Specify your local GPU or CPU.
        )

        # Make a forecast on the target column given the input data.
        future_forecast = pipeline(historical_df, future_time_series=controls_df)
        future_forecast.tail()

        # Pre-cast the timestamp to avoid future dtype inference changes
        historical_df[timestamp_column] = pd.to_datetime(historical_df[timestamp_column])

        # Plot the historical data and predicted series.
        plot_predictions(
            input_df=historical_df,
            predictions_df=future_forecast,
            freq=tsp.freq,
            timestamp_column=timestamp_column,
            channel=target_columns[0],
        )


    plt.show()
   
if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_model()
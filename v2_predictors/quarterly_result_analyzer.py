import argparse

import torch
import pandas as pd
import numpy
import random

import fitz
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lightgbm as lgb
import pickle

from edgar import *
from tiingo import TiingoClient
import yfinance as yf

STOCK_PRICE_CHANGE_IN_PERCENTS_ON_REPORT_DAY_COLUMN = "price_change_in_percent"
FINANCIALS_REPORT_LABEL_COLUMN = "concept"
EARNINGS_REPORT_DATE = ""
EMPTY_ROWS_IN_DEFAULT_EPS_REPORT = 4
EPS_DIFF_COLUMN_NAME = 'Surprise(%)'

parser = argparse.ArgumentParser("stock_predict")
parser.add_argument('--email', required=True, help='User email address for accessing Edgar data for financials analysis')
parser.add_argument('--ticker', required=True, help='Stock ticker to analyze')
parser.add_argument('--api_key', help='NOTE: Required if --retrain_financials_analysis is used. Tiingo API key for stock price query')
parser.add_argument('--quarterly_report_file', help='NOTE: Required if --run_sentiment_analysis is used. The report file to analyze with sentiment analysis')
parser.add_argument('--run_sentiment_analysis', dest='run_sentiment_analysis', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--run_financials_analysis', dest='run_financials_analysis', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--retrain_financials_analysis', dest='retrain_financials_analysis', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--load_saved_traning_data', dest='load_saved_traning_data', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--save_training_data_to_file', dest='save_training_data_to_file', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

def getPriceChangeOnReportDay(ticker, report_date_str, api_key):
    print(f"Getting stock price data for: {ticker} for date: {report_date_str}")
    earliest_possible_day_delta_for_pre_earnings_data_for_before_market_open_report = 3
    latest_possible_day_delta_for_post_earnings_data_for_after_market_close_report = 3
    report_date = pd.to_datetime(report_date_str)
    start_date = (report_date - pd.Timedelta(days=earliest_possible_day_delta_for_pre_earnings_data_for_before_market_open_report)).strftime('%Y-%m-%d')
    end_date = (report_date + pd.Timedelta(days=latest_possible_day_delta_for_post_earnings_data_for_after_market_close_report)).strftime('%Y-%m-%d')

    config = {
    'api_key': api_key
}
    client = TiingoClient(config)
    hist_price_array = client.get_ticker_price(ticker, startDate=start_date, endDate=end_date, frequency='daily')
    # In case of the before market open report, we will get closing price of the previous day
    # and in case of the after market open report, we will get closing price of next day
    pre_earnings_price = None
    post_earnings_price = None
    for i in range(1, earliest_possible_day_delta_for_pre_earnings_data_for_before_market_open_report + 1):
        date_to_check = (report_date - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        price_info = next((item for item in hist_price_array if item['date'].startswith(date_to_check)), None)
        if price_info is not None:
            pre_earnings_price = price_info['close']
        else:
            print(f"No pre earnings price information found for {date_to_check}")
    
    for i in range(1, latest_possible_day_delta_for_post_earnings_data_for_after_market_close_report + 1):
        date_to_check = (report_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        price_info = next((item for item in hist_price_array if item['date'].startswith(date_to_check)), None)
        if price_info is not None:
            post_earnings_price = price_info['close']
        else:
            print(f"No post earnings price information found for {date_to_check}")
        
    if not pre_earnings_price or not post_earnings_price:
        print("Could not find a price data for required days")
        return None

    price_change = post_earnings_price - pre_earnings_price
    percent_change = (price_change / pre_earnings_price) * 100
    
    print(f"Pre-earnings price: {pre_earnings_price}")
    print(f"Post-earnings price: {post_earnings_price}")
    print(f"Price Change: {price_change:.2f} USD ({percent_change:.2f}%)")
    return percent_change

# Get dataframe with columns:
# "concept": which contains "label" values like us-gaap_ProfitLoss, us-gaap_NetIncomeLoss
# report date e.g. "2025-03-31": which contains reported values for the period,
# In 10-Q income and cash flow reports, the column name contains quarter postfix e.g. "2025-03-31 Q1",
# within this function we wil rename it to date without quarter postfix i.e. to format like e.g. "2025-03-31"
def getDataFrameWithOnlyRelevantColumns(dataframe, earnings_date, earnings_date_with_postfix):
    try:
        return dataframe[[FINANCIALS_REPORT_LABEL_COLUMN, earnings_date]]
    except:
        try:
            df_with_relevant_columns = dataframe[[FINANCIALS_REPORT_LABEL_COLUMN, earnings_date_with_postfix]]
            return df_with_relevant_columns.rename(columns={earnings_date_with_postfix: earnings_date})
        except:
            print(f"Failed to find correct earnings date {earnings_date} on dataframe, returning empty dataframe")
            return pd.DataFrame()
           
def getEPSSurpiseData(ticker, number_of_quarters):
    stock = yf.Ticker(ticker)
    return stock.get_earnings_dates(limit=(number_of_quarters + EMPTY_ROWS_IN_DEFAULT_EPS_REPORT)).iloc[EMPTY_ROWS_IN_DEFAULT_EPS_REPORT:].fillna(0)
    
def parseDataFrameFromFinancials(ticker, report_count, retrain, api_key, email):
    set_identity(email)
    company = Company(ticker)
    df_final_combined_statements = pd.DataFrame()
    filings = company.get_filings()
    financials_filings = filings.filter(form=['10-K', '10-Q'])
    if len(financials_filings) > report_count:
        for i in range(report_count):
            if ((i + 4) <= report_count):
                financials_obj_current = financials_filings[i].data_object()
                financials_obj_to_compare_to = financials_filings[i + 4].data_object()
                prev_df_combined_statements = None
                compare_to_prev = False
                price_change = None
                for financials_obj in [financials_obj_current, financials_obj_to_compare_to]:
                    df_income_statement_raw = financials_obj.income_statement.to_dataframe()
                    latest_earnings_date_column_raw_name = df_income_statement_raw.columns[2]
                    latest_earnings_date_column_sanitized_name = latest_earnings_date_column_raw_name.split(" ", 1)[0] # In 10-Q income and cash flow reports, the column name contains quarter postfix e.g. "2025-03-31 Q1"  
                    actual_earnings_release_date = financials_obj.filing_date
                    df_income_statement = getDataFrameWithOnlyRelevantColumns(df_income_statement_raw, latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
                    #NOTE: including the balance sheet and cash flow statement to data set will cause noise and importantance of features such as us-gaap_EarningsPerShareBasic is "lost"
                    #df_balance_sheet = getDataFrameWithOnlyRelevantColumns(financials_obj.balance_sheet.to_dataframe(), latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
                    #df_cash_flow_statement = getDataFrameWithOnlyRelevantColumns(financials_obj.cash_flow_statement.to_dataframe(), latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
                 
                    df_quarterly_financials_combined = pd.concat([df_income_statement])
                    
                    # transpose data frame to have data "features" as columns, set concept as label column as concept contains items as us-gaap_ProfitLoss, us-gaap_NetIncomeLoss.., 
                    # add suffix to duplicate columns
                    df_quarterly_financials_combined = df_quarterly_financials_combined.transpose()
                    df_quarterly_financials_combined.columns = df_quarterly_financials_combined.iloc[0]
                    new_columns = pd.Series(df_quarterly_financials_combined.columns).groupby(df_quarterly_financials_combined.columns).cumcount().astype(str)
                    df_quarterly_financials_combined.columns = [f"{col}_{suffix}" if suffix != "0" else col for col, suffix in zip(df_quarterly_financials_combined.columns, new_columns)]
                    # calculate the diff of latest report financials and previous report financials
                    df_quarterly_financials_combined = df_quarterly_financials_combined.apply(pd.to_numeric, errors="coerce")

                    if retrain and not compare_to_prev: # we only fetch price change if we are retraining with historical data, in case of prediction, we won't have price data available
                        price_change = getPriceChangeOnReportDay(args.ticker, actual_earnings_release_date, api_key)
                    # Add row of comparing YoY financials to final dataframe. 
                    # Keep only the row with numeric results diff, replace NAN values with 0
                    if compare_to_prev:
                        df_diffed_financials = pd.DataFrame(prev_df_combined_statements.iloc[1] - df_quarterly_financials_combined.iloc[1]).transpose()
                        if retrain:
                            df_diffed_financials[STOCK_PRICE_CHANGE_IN_PERCENTS_ON_REPORT_DAY_COLUMN] = price_change
                        df_final_combined_statements = pd.concat([df_final_combined_statements, df_diffed_financials], ignore_index=True)
                    else:
                        prev_df_combined_statements = df_quarterly_financials_combined
                        global EARNINGS_REPORT_DATE
                        EARNINGS_REPORT_DATE = actual_earnings_release_date
                        compare_to_prev = True
        df_eps_surprise_data = getEPSSurpiseData(ticker, report_count - 3)
        print(f"DEBUG_KERKJO: Check df_final_combined_statements shape: {df_final_combined_statements.shape}, check df_eps_surprise_data shape: {df_eps_surprise_data.shape}")
        df_final_combined_statements[EPS_DIFF_COLUMN_NAME] = df_eps_surprise_data[EPS_DIFF_COLUMN_NAME].values
        df_final_combined_statements.fillna(0, inplace=True)
        df_only_numeric = df_final_combined_statements.apply(pd.to_numeric, errors="coerce")
        print(f"Check cleaned finalized financials data for training or prediction: {df_only_numeric}")
        return df_only_numeric
    else:
        print(f"Didn't find enough financial filings for {ticker}. Needed {report_count} filings, found {len(financials_filings)} filings.")
    return pd.DataFrame()
    
def getFinancialsDataOnlyNumeric(email, ticker, retrain, api_key):
    if retrain:
        print(f"Getting last 12 financial data reports for: {ticker}")
        return parseDataFrameFromFinancials(ticker, 12, retrain, api_key, email)
    else:    
        print(f"Getting last financial data report for: {ticker}")
        return parseDataFrameFromFinancials(ticker, 4, retrain, api_key, email)   
    
def saveFinancialsModel(model, ticker, model_name="lightgbm_financials_predict_model"):
    with open(model_name + ticker + ".pkl", "wb") as f:
        pickle.dump(model, f)
    
def loadFinancialsModel(ticker, model_name="lightgbm_financials_predict_model"):    
    with open(model_name + ticker + ".pkl", "rb") as f:
        return pickle.load(f)

def main():    
    if args.run_sentiment_analysis and args.quarterly_report_file:
        doc = fitz.open(args.quarterly_report_file)
        text = "\n".join([page.get_text() for page in doc])
        df = pd.DataFrame([text], columns=["Extracted_Text"])
        X = df['Extracted_Text'].to_list()

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        preds = []
        preds_proba = []
        tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
        for x in X:
            with torch.no_grad():
                input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
                logits = model(**input_sequence).logits
                scores = {
                k: v
                for k, v in zip(
                    model.config.id2label.values(),
                    scipy.special.softmax(logits.numpy().squeeze()),
                )
            }
            sentimentFinbert = max(scores, key=scores.get)
            probabilityFinbert = max(scores.values())
            preds.append(sentimentFinbert)
            preds_proba.append(probabilityFinbert)
            
        print(f'Predictions: {preds}, probabilities: {preds_proba}')

    if args.run_financials_analysis or args.retrain_financials_analysis:
        api_key = '' if not args.api_key else args.api_key
        df_combined_financials_only_numeric_data = pd.DataFrame()
        if args.load_saved_traning_data:
            df_combined_financials_only_numeric_data = pd.read_csv(f"{args.ticker}_traning_data.csv")
        else:
            df_combined_financials_only_numeric_data = getFinancialsDataOnlyNumeric(args.email, args.ticker, args.retrain_financials_analysis, api_key)
            if not df_combined_financials_only_numeric_data.empty and args.save_training_data_to_file:
                df_combined_financials_only_numeric_data.to_csv(f"{args.ticker}_traning_data.csv", index=False)
        if df_combined_financials_only_numeric_data.empty:
            print(f"Unable to get traning data for {args.ticker}")
            return
        model = None
        if args.retrain_financials_analysis:
            columns = list(df_combined_financials_only_numeric_data.columns)
            columns[-2], columns[-1] = columns[-1], columns[-2]
            df_combined_financials_only_numeric_data = df_combined_financials_only_numeric_data[columns]
            X = df_combined_financials_only_numeric_data.iloc[:, :-1] # features, all columns except the price change column
            y = df_combined_financials_only_numeric_data.iloc[:, -1] # target column

            params = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'metric': 'rmse',
                'num_leaves': 2,
                'max_depth': 2,
                'learning_rate': 0.03,
                'n_estimators': 100,
                'min_child_samples': 1,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'subsample': 0.8,
                'subsample_freq': 1,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 15
            }
            model = lgb.LGBMRegressor(**params)
            loo = LeaveOneOut()
            validation_rmse_scores = []
            train_rmse_scores = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index] 
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric="rmse")
                
                train_rmse = model.evals_result_["training"]["rmse"][-1]
                validation_rmse = model.evals_result_["valid_1"]["rmse"][-1]
                
                train_rmse_scores.append(train_rmse)
                validation_rmse_scores.append(validation_rmse)

            feature_importances = model.feature_importances_ 
            feature_names = model.booster_.feature_name()
            importance_df = pd.DataFrame({ 'feature': feature_names, 'importance': feature_importances }) 
            importance_df.to_csv(f"{args.ticker}_feature_importance_data.csv", index=False)
            print(f"Check feature importance from: {args.ticker}_feature_importance_data.csv")
            
            #TODO: we have overfitting problem: Average training RMSE: 0.49, Average validation RMSE: 3.15
            print(f"Model training RMSE scores: {train_rmse_scores}")
            print(f"Average training RMSE: {numpy.mean(train_rmse_scores)}")
            print(f"Model validation RMSE scores: {validation_rmse_scores}")
            print(f"Average validation RMSE: {numpy.mean(validation_rmse_scores)}")

            saveFinancialsModel(model, args.ticker)
        else:
            model = loadFinancialsModel(args.ticker)
            final_pred = model.predict(df_combined_financials_only_numeric_data, predict_disable_shape_check=True)
            print(f"Predicted value for {args.ticker} price change in percents on earnings report release date for: {EARNINGS_REPORT_DATE} is: {final_pred}")
        
if __name__ == "__main__":
    main()
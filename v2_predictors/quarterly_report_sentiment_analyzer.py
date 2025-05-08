import argparse

import torch
import pandas as pd
import numpy
import random

import fitz
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lightgbm as lgb
import pickle

from edgar import *
from tiingo import TiingoClient

#TODO: 
# - Add quarterly report and analyze it with pre-trained finBERT (DONE)
# - Get the stock percentage change on Quartely report days and input to LightGBM (DONE)
# - Evaluate and refine LightGBM training and params
 

STOCK_PRICE_CHANGE_IN_PERCENTS_ON_REPORT_DAY_COLUMN = "price_change_in_percent"
FINANCIALS_REPORT_LABEL_COLUMN = "concept"

parser = argparse.ArgumentParser("stock_predict")
parser.add_argument('--email', required=True, help='User email address for accessing Edgar data for financials analysis')
parser.add_argument('--ticker', required=True, help='Stock ticker to analyze')
parser.add_argument('--api_key', help='NOTE: Required if --retrain_financials_analysis is used. Tiingo API key for stock price query')
parser.add_argument('--quarterly_report_file', help='NOTE: Required if --run_sentiment_analysis is used. The report file to analyze with sentiment analysis')
parser.add_argument('--run_sentiment_analysis', dest='run_sentiment_analysis', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--run_financials_analysis', dest='run_financials_analysis', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--retrain_financials_analysis', dest='retrain_financials_analysis', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

def get_price_change_on_report_day(ticker, report_date_str, api_key):
    print(f"Getting stock price data for: {ticker} for date: {report_date_str}")
    earliest_possible_day_delta_for_pre_earnings_data_for_before_market_open_report = 4
    latest_possible_day_delta_for_post_earnings_data_for_after_market_close_report = 4
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

# Get data frame with columns:
# "concept": which contains "label" values like us-gaap_ProfitLoss, us-gaap_NetIncomeLoss
# report date e.g. "2025-03-31": which contains reported values for the period,
# In 10-Q income and cash flow reports, the column name contains quarter postfix e.g. "2025-03-31 Q1",
# within this function we wil rename it to date without quarter postfix i.e. to format like e.g. "2025-03-31"
def getDataFrameWithOnlyRelevantColumns(dataframe, earnings_date, earnings_date_with_postfix):
    dataframe.to_csv(f"some_dataframe_{earnings_date}.csv", index=True)
    try:
        return dataframe[[FINANCIALS_REPORT_LABEL_COLUMN, earnings_date]]
    except:
        try:
            df_with_relevant_columns = dataframe[[FINANCIALS_REPORT_LABEL_COLUMN, earnings_date_with_postfix]]
            return df_with_relevant_columns.rename(columns={earnings_date_with_postfix: earnings_date})
        except:
            print(f"Failed to find correct earnings date {earnings_date} on dataframe, returning empty dataframe")
            return pd.DataFrame()

def parseDataFrameFromFinancials(company, report_count, retrain, api_key):
    prev_df_combined_statements = None
    df_final_combined_statements = pd.DataFrame()
    filings = company.get_filings()
    financials_filings = filings.filter(form=['10-K', '10-Q'])
    for i in range(report_count):
        financials_obj = financials_filings[i].data_object()        
        df_income_statement_raw = financials_obj.income_statement.to_dataframe()
        latest_earnings_date_column_raw_name = df_income_statement_raw.columns[2]
        latest_earnings_date_column_sanitized_name = latest_earnings_date_column_raw_name.split(" ", 1)[0] # In 10-Q income and cash flow reports, the column name contains quarter postfix e.g. "2025-03-31 Q1"  
        
        df_income_statement = getDataFrameWithOnlyRelevantColumns(df_income_statement_raw, latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
        df_balance_sheet = getDataFrameWithOnlyRelevantColumns(financials_obj.balance_sheet.to_dataframe(), latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
        df_cash_flow_statement = getDataFrameWithOnlyRelevantColumns(financials_obj.cash_flow_statement.to_dataframe(), latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
     
        df_quarterly_financials_combined = pd.concat([df_cash_flow_statement, df_balance_sheet, df_income_statement])
        
        # transpose data frame to have data "features" as columns, set concept as label column as concept contains items as us-gaap_ProfitLoss, us-gaap_NetIncomeLoss.., 
        # add suffix to duplicate columns
        df_quarterly_financials_combined = df_quarterly_financials_combined.transpose()
        df_quarterly_financials_combined.columns = df_quarterly_financials_combined.iloc[0]
        new_columns = pd.Series(df_quarterly_financials_combined.columns).groupby(df_quarterly_financials_combined.columns).cumcount().astype(str)
        df_quarterly_financials_combined.columns = [f"{col}_{suffix}" if suffix != "0" else col for col, suffix in zip(df_quarterly_financials_combined.columns, new_columns)]
        # calculate the diff of latest report financials and previous report financials
        df_quarterly_financials_combined = df_quarterly_financials_combined.apply(pd.to_numeric, errors="coerce")

        price_change = None
        if retrain: # we only fetch price change if we are retraining with historical data, in case of prediction, we won't have price data available
            price_change = get_price_change_on_report_day(args.ticker, latest_earnings_date_column_sanitized_name, api_key)
        # Keep only the row with numeric results diff, replace NAN values with 0
        if i > 0 and i < report_count:
            df_diffed_financials = pd.DataFrame(prev_df_combined_statements.iloc[1] - df_quarterly_financials_combined.iloc[1]).transpose()
            if price_change:
                df_diffed_financials[STOCK_PRICE_CHANGE_IN_PERCENTS_ON_REPORT_DAY_COLUMN] = price_change
            df_final_combined_statements = pd.concat([df_final_combined_statements, df_diffed_financials], ignore_index=True)
            prev_df_combined_statements = df_quarterly_financials_combined
        else:
            prev_df_combined_statements = df_quarterly_financials_combined
            if price_change:
                prev_df_combined_statements[STOCK_PRICE_CHANGE_IN_PERCENTS_ON_REPORT_DAY_COLUMN] = price_change
    df_final_combined_statements.fillna(0, inplace=True)
    df_only_numeric = df_final_combined_statements.apply(pd.to_numeric, errors="coerce")
    print(f"Check cleaned finalized financials data for training or prediction: {df_only_numeric}")
    df_only_numeric.to_csv(f"df_only_numeric.csv", index=True)
    return df_only_numeric
    
def getFinancialsDataOnlyNumeric(email, ticker, retrain, api_key):
    set_identity(email)
    company = Company(ticker)
    if retrain:
        print(f"Getting last 12 financial data reports for: {ticker}")
        return parseDataFrameFromFinancials(company, 13, retrain, api_key)
    else:    
        print(f"Getting last financial data report for: {ticker}")
        return parseDataFrameFromFinancials(company, 2, retrain, api_key)   
    
def save_financials_model(model, ticker, model_name="lightgbm_financials_predict_model"):
    with open(model_name + ticker + ".pkl", "wb") as f:
        pickle.dump(model, f)
    
def load_financials_model(ticker, model_name="lightgbm_financials_predict_model"):    
    with open(model_name + ticker + ".pkl", "rb") as f:
        return pickle.load(f)

def main():    
    if args.run_sentiment_analysis and quarterly_report_file:
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
        df_combined_financials_only_numeric_data = getFinancialsDataOnlyNumeric(args.email, args.ticker, args.retrain_financials_analysis, api_key)
        model = None
        if args.retrain_financials_analysis:
            X = df_combined_financials_only_numeric_data.iloc[:, 1:] # features, all columns except the price change column
            y = df_combined_financials_only_numeric_data.iloc[:, 0] # target column
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'regression', 
                'metric': 'mse',  
                'boosting_type': 'gbdt',
                'verbose': -1
            }
            
            model = lgb.train(params, train_data, num_boost_round=100)
            print(f"LightGBM model importance value: {model.feature_importance()}")
            y_pred = model.predict(X_test)
            print(f"LightGBM model test predict value: {y_pred}")
            model.save_model("lightgbm_model.txt")
            save_financials_model(model, args.ticker)
        else:
            model = load_financials_model(args.ticker)
            final_pred = model.predict(df_combined_financials_only_numeric_data, predict_disable_shape_check=True)
            print(f"LightGBM model test predict value: {final_pred}")
        
if __name__ == "__main__":
    main()
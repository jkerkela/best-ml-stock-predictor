import argparse
from enum import Enum
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from edgar import *
import yfinance as yf

import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

import optuna


parser = argparse.ArgumentParser("company_fundamentals_analysis")
parser.add_argument('--email', required=True, help='User email address for accessing Edgar data for financials analysis')
parser.add_argument('--ticker', required=True, help='Stock ticker to analyze')
parser.add_argument('--save_data_to_disk', dest='save_data_to_disk', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--load_data_from_disk', dest='load_data_from_disk', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

FIVE_YEARS_OF_REPORTS = 20
FINANCIALS_REPORT_ORIGINAL_LABEL_COLUMN = "concept"
INCOME_STATEMENT_OPERATING_MARGIN_COLUMN = "operating_margin"
INCOME_STATEMENT_PE_COLUMN = "PE"
BALANCE_SHEET_TANGIBLE_BOOK_VALUE_COLUMN = "tangible_book_value"
BALANCE_SHEET_DEBT_TO_ASSETS_RATIO_COLUMN = "debt_to_assets_ratio"
CASHFLOW_STATEMENT_FREE_CASHFLOW_COLUMN = "free_cashdflow"
STOCK_PRICE_COLUMN = "stock_price"

RELEVANT_INCOME_STATEMENT_COLUMNS = [
        STOCK_PRICE_COLUMN,
        'us-gaap_Revenues',
        'us-gaap_OperatingExpenses', 
        'us-gaap_ResearchAndDevelopmentExpense', 
        'us-gaap_GrossProfit',
        'us-gaap_OperatingIncomeLoss',
        'us-gaap_EarningsPerShareBasic',
        INCOME_STATEMENT_OPERATING_MARGIN_COLUMN,
        INCOME_STATEMENT_PE_COLUMN
    ]
    
RELEVANT_BALANCE_SHEET_COLUMNS = [
        'us-gaap_LongTermDebtNoncurrent',
        'us-gaap_StockholdersEquity', 
#        'us-gaap_Liabilities', 
#        'us-gaap_Assets',
#        'us-gaap_IntangibleAssetsNetExcludingGoodwill',
        BALANCE_SHEET_TANGIBLE_BOOK_VALUE_COLUMN,
        BALANCE_SHEET_DEBT_TO_ASSETS_RATIO_COLUMN
    ]
    
RELEVANT_CASH_FLOW_STATEMENT_COLUMNS = [
        'us-gaap_NetCashProvidedByUsedInOperatingActivities',
        CASHFLOW_STATEMENT_FREE_CASHFLOW_COLUMN
#        'us-gaap_NetCashProvidedByUsedInFinancingActivities',
#        'us-gaap_NetCashProvidedByUsedInInvestingActivities'
    ]



class DocumentType(Enum):
    INCOME_STATEMENT = 1
    BALANCE_SHEET = 2
    CASHFLOW_STATEMENT = 3
    
def is_datetime(string, format="%Y-%m-%d"):
    try:
        datetime.strptime(string, format)
        return True
    except ValueError:
        print(f"DEBUG: invalid format date time string: {string}")
        return False

# Get dataframe with columns:
# "concept": which contains "label" values like us-gaap_ProfitLoss, us-gaap_NetIncomeLoss
# report date e.g. "2025-03-31": which contains reported values for the period,
# In 10-Q income and cash flow reports, the column name contains quarter postfix e.g. "2025-03-31 Q1",
# within this function we wil rename it to date without quarter postfix i.e. to format like e.g. "2025-03-31"
def getDataFrameWithOnlyLatestData(dataframe, earnings_date, earnings_date_with_postfix):
    try:
        df_with_relevant_columns = dataframe[[FINANCIALS_REPORT_ORIGINAL_LABEL_COLUMN, earnings_date_with_postfix]]
        return df_with_relevant_columns.rename(columns={earnings_date_with_postfix: earnings_date})
    except:
        try:
            return dataframe[[FINANCIALS_REPORT_ORIGINAL_LABEL_COLUMN, earnings_date]]
        except:
            print(f"Failed to find correct earnings date {earnings_date} on dataframe, returning empty dataframe")
            return pd.DataFrame()

def getWithTransformFeaturesToColumns(dataframe):
    dataframe = dataframe.transpose()
    dataframe.columns = dataframe.iloc[0]
    dataframe.drop(FINANCIALS_REPORT_ORIGINAL_LABEL_COLUMN, inplace=True)
    dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
    return dataframe

def getSharePrice(ticker, date):
    print(f"Getting stock price data for: {ticker} for date: {date}")
    price_on_date = 0
    earliest_possible_day_delta_for_pre_earnings_data_for_before_market_open_report = 3
    latest_possible_day_delta_for_post_earnings_data_for_after_market_close_report = 3
    report_date = pd.to_datetime(date)
    start_date = (report_date - pd.Timedelta(days=earliest_possible_day_delta_for_pre_earnings_data_for_before_market_open_report)).strftime('%Y-%m-%d')
    end_date = (report_date + pd.Timedelta(days=latest_possible_day_delta_for_post_earnings_data_for_after_market_close_report)).strftime('%Y-%m-%d')

    stock = yf.Ticker(ticker)
    history = stock.history(start=start_date, end=end_date)

    if not history.empty:
        price_on_date = history.iloc[0]['Close']
        print(f"Stock price on {date}: {price_on_date}")
    else:
        print(f"No data available for {date}")
        
    return price_on_date

def getWithNeededColumnsCalculated(dataframe, document_type, share_price=0):
    if document_type is DocumentType.INCOME_STATEMENT:
        if set(('us-gaap_Revenues', 'us-gaap_OperatingIncomeLoss')).issubset(dataframe.columns):
            dataframe[INCOME_STATEMENT_OPERATING_MARGIN_COLUMN] = dataframe['us-gaap_OperatingIncomeLoss'] / dataframe['us-gaap_Revenues']
        if 'us-gaap_EarningsPerShareBasic' in dataframe.columns and share_price > 0:
            dataframe[INCOME_STATEMENT_PE_COLUMN] = share_price / dataframe['us-gaap_EarningsPerShareBasic']
            dataframe[STOCK_PRICE_COLUMN] = share_price
    elif document_type is DocumentType.BALANCE_SHEET:
        if set(('us-gaap_Assets', 'us-gaap_Liabilities')).issubset(dataframe.columns):
            dataframe[BALANCE_SHEET_DEBT_TO_ASSETS_RATIO_COLUMN] = dataframe['us-gaap_Liabilities'] / dataframe['us-gaap_Assets']
            if 'gaap_IntangibleAssetsNetExcludingGoodwill' in dataframe.columns:
                dataframe[BALANCE_SHEET_TANGIBLE_BOOK_VALUE_COLUMN] = dataframe['us-gaap_Assets'] - dataframe['us-gaap_Liabilities'] - dataframe['us-gaap_IntangibleAssetsNetExcludingGoodwill']
    elif document_type is DocumentType.CASHFLOW_STATEMENT:
        if set(('us-gaap_NetCashProvidedByUsedInOperatingActivities', 'us-gaap_NetCashProvidedByUsedInFinancingActivities', 'us-gaap_NetCashProvidedByUsedInInvestingActivities')).issubset(dataframe.columns):
            dataframe[CASHFLOW_STATEMENT_FREE_CASHFLOW_COLUMN] = dataframe['us-gaap_NetCashProvidedByUsedInOperatingActivities'] + dataframe['us-gaap_NetCashProvidedByUsedInFinancingActivities'] + dataframe['us-gaap_NetCashProvidedByUsedInInvestingActivities']
    return dataframe

def getWithUnnecessaryColumnsFiltered(dataframe, document_type):
    if document_type is DocumentType.INCOME_STATEMENT:
        return dataframe[dataframe.columns.intersection(RELEVANT_INCOME_STATEMENT_COLUMNS)]
    elif document_type is DocumentType.BALANCE_SHEET:
        return dataframe[dataframe.columns.intersection(RELEVANT_BALANCE_SHEET_COLUMNS)]
    elif document_type is DocumentType.CASHFLOW_STATEMENT:
        return dataframe[dataframe.columns.intersection(RELEVANT_CASH_FLOW_STATEMENT_COLUMNS)]
    else:
        return dataframe
        

def saveDataToDisk(dataframe, ticker):
    dataframe.to_csv(f"{ticker}_fundamental_data.csv", index=True)
    
def loadDataFromDisk(ticker):
    return pd.read_csv(f"{ticker}_fundamental_data.csv", index_col=0)
    
def fetchCompanyFundamentals(email, ticker, report_count=FIVE_YEARS_OF_REPORTS):
#		- financial statement, balance sheet and cashflow statement
#			Recheck for relevance everything included below
#			- from financial analyse:
#				- Revenue:  us-gaap_Revenues
#				- Operating expenses, especially R&D: us-gaap_OperatingExpenses,  us-gaap_ResearchAndDevelopmentExpense
#				- Gross profit (?): us-gaap_GrossProfit
#				- Operating income: us-gaap_OperatingIncomeLoss
#				- EPS: us-gaap_EarningsPerShareBasic
#				- calculate: operating profit margin:  Operating Income / Revenue
#				- calculate: PE: company: Market Price per Share / Earnings per Share
#
#			- from balance sheet analyse:
#				- tangible book value:  us-gaap_Assets - us-gaap_Liabilities - us-gaap_IntangibleAssetsNetExcludingGoodwill
#				- total debt: us-gaap_LongTermDebtNoncurrent
#				- shareholders equity: us-gaap_StockholdersEquity
#				- calculate: Debt-to-assets ratio: Total Liabilities / Total Assets: us-gaap_Liabilities, us-gaap_Assets
#
#			- from cash flow statement:
#				- free cash flow us-gaap_NetCashProvidedByUsedInOperatingActivities + us-gaap_NetCashProvidedByUsedInFinancingActivities + us-gaap_NetCashProvidedByUsedInInvestingActivities
#				- operating cash flow us-gaap_NetCashProvidedByUsedInOperatingActivities
    set_identity(email)
    company = Company(ticker)
    df_final_combined_statements = pd.DataFrame()
    filings = company.get_filings()
    financials_filings = filings.filter(form=['10-K', '10-Q'])
    available_reports = min(len(financials_filings), report_count) 
    for i in range(available_reports):
        financials_obj = financials_filings[i].data_object()
        df_income_statement_raw = financials_obj.income_statement.to_dataframe()
        latest_earnings_date_column_raw_name = df_income_statement_raw.columns[2]
        latest_earnings_date_column_sanitized_name = latest_earnings_date_column_raw_name.split(" ", 1)[0] # In 10-Q income and cash flow reports, the column name contains quarter postfix e.g. "2025-03-31 Q1"  
        if is_datetime(latest_earnings_date_column_sanitized_name):
            print(f"Check latest_earnings_date_column_sanitized_name {latest_earnings_date_column_sanitized_name} ")
            share_price = getSharePrice(ticker, latest_earnings_date_column_sanitized_name)
            df_income_statement = getDataFrameWithOnlyLatestData(df_income_statement_raw, latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
            if not df_income_statement.empty:
                df_income_statement = getWithTransformFeaturesToColumns(df_income_statement)
                df_income_statement = getWithNeededColumnsCalculated(df_income_statement, DocumentType.INCOME_STATEMENT, share_price)
                df_income_statement = getWithUnnecessaryColumnsFiltered(df_income_statement, DocumentType.INCOME_STATEMENT)
            
            df_balance_sheet_raw = financials_obj.balance_sheet.to_dataframe()
            df_balance_sheet = getDataFrameWithOnlyLatestData(df_balance_sheet_raw, latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
            if not df_balance_sheet.empty:
                df_balance_sheet = getWithTransformFeaturesToColumns(df_balance_sheet)
                df_balance_sheet = getWithNeededColumnsCalculated(df_balance_sheet, DocumentType.BALANCE_SHEET)
                df_balance_sheet = getWithUnnecessaryColumnsFiltered(df_balance_sheet, DocumentType.BALANCE_SHEET)
            
            df_cashflow_statement_raw = financials_obj.cash_flow_statement.to_dataframe()
            df_cashflow_statement = getDataFrameWithOnlyLatestData(df_cashflow_statement_raw, latest_earnings_date_column_sanitized_name, latest_earnings_date_column_raw_name)
            if not df_cashflow_statement.empty:
                df_cashflow_statement = getWithTransformFeaturesToColumns(df_cashflow_statement)
                df_cashflow_statement = getWithNeededColumnsCalculated(df_cashflow_statement, DocumentType.CASHFLOW_STATEMENT)
                df_cashflow_statement = getWithUnnecessaryColumnsFiltered(df_cashflow_statement, DocumentType.CASHFLOW_STATEMENT)

            df_statements_combined = pd.concat([df_income_statement, df_balance_sheet, df_cashflow_statement], axis=1)
            
            df_final_combined_statements = pd.concat([df_final_combined_statements, df_statements_combined], axis=0, ignore_index=False)

    if df_final_combined_statements.empty:
        print(f"Didn't find financial filings for {ticker}")
        return pd.DataFrame()
    df_final_combined_statements.fillna(0, inplace=True)
    df_final_combined_statements = df_final_combined_statements[::-1]
    print(f"Check cleaned finalized cash flow statement: {df_final_combined_statements} \n check index: {df_final_combined_statements.index}")
    return df_final_combined_statements


def chartCompanyFundamentals(dataframe):
    for col in dataframe.columns:
        plt.figure(figsize=(30, 10))
        plt.plot(dataframe[col], marker='o', linestyle='-')
        plt.title(f'Column: {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.grid()
        plt.savefig(f"{col}_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        

def objective(trial, X, y):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2)
    }
    
    model = XGBRegressor(objective="reg:squarederror", **params)
    score = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    return score.mean()
    
def evaluateParams(dataframe):
    X = dataframe.drop(columns=[STOCK_PRICE_COLUMN])
    y = dataframe[STOCK_PRICE_COLUMN] 
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    print("Best Parameters:", study.best_params)
    return study.best_params
    
def trainModel(dataframe, params): 
    loo = LeaveOneOut()
    X = dataframe.drop(columns=[STOCK_PRICE_COLUMN])
    y = dataframe[STOCK_PRICE_COLUMN] 
    feature_importance_list = []
    cv_scores = []
    
    params['objective'] = "reg:squarederror"
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        feature_importance_list.append(model.get_booster().get_score(importance_type="gain"))
        y_pred = model.predict(X_test)
        cv_scores.append(mean_absolute_error(y_test, y_pred))

    avg_importance = {}
    for importance_dict in feature_importance_list:
        for feature, value in importance_dict.items():
            avg_importance[feature] = avg_importance.get(feature, 0) + value
    for feature in avg_importance:
        avg_importance[feature] /= len(feature_importance_list)
    avg_cv_score = np.mean(cv_scores)

    print("Average Feature Importance Across LOOCV:", avg_importance)
    print(f"Average Cross-Validation Score (MAE): {avg_cv_score}")

def main():
    df_company_fundamentals = pd.DataFrame()
    if args.load_data_from_disk:    
        df_company_fundamentals = loadDataFromDisk(args.ticker)
    else:
        df_company_fundamentals = fetchCompanyFundamentals(args.email, args.ticker)
        if args.save_data_to_disk:
            saveDataToDisk(df_company_fundamentals, args.ticker)
    chartCompanyFundamentals(df_company_fundamentals)
    
    params = evaluateParams(df_company_fundamentals)
    trainModel(df_company_fundamentals, params)

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#TODO: change to use data set downloaded from
# https://www.kaggle.com/datasets/bhadramohit/bitcoin-datasetintervals-of-1-minute?resource=download
TRAINING_DATA_FILE_PATH = "btcusd_1-min_data2.csv"
timestamp_column = "Timestamp"

SMI_LABEL = 'SMI'
MACD_LABEL= 'MACD'
MACD_SIGNAL_LABEL = 'MACD_SIGNAL'

minumum_stop_distance_in_percents = 0.002
needed_price_increase_factor = 1 + minumum_stop_distance_in_percents

def calculate_smi(df, period=14):
    df['Lowest_Low'] = df['Close'].rolling(window=period).min()
    df['Highest_High'] = df['Close'].rolling(window=period).max()
    df['Midpoint'] = (df['Highest_High'] + df['Lowest_Low']) / 2
    
    # Calculate SMI
    df[SMI_LABEL] = 100 * ((df['Close'] - df['Midpoint']) / (df['Highest_High'] - df['Lowest_Low']))
    
def calculate_macd(df)
    # Calculate MACD line (difference between short-term and long-term EMAs)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df[MACD_LABEL] = ema12 - ema26
    # Calculate Signal Line (9-period EMA of MACD)
    df[MACD_SIGNAL_LABEL] = df[MACD_LABEL].ewm(span=9, adjust=False).mean()
    
def main():   
    data = pd.read_csv(TRAINING_DATA_FILE_PATH)
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])

    calculate_smi(data)
    calculate_macd(data)
    #print(data[['Close', 'MACD', 'Signal_Line']])
    # Find crossovers
    data['Bullish_Crossover'] = (data[MACD_LABEL] > data[MACD_SIGNAL_LABEL]) & (data[MACD_LABEL].shift(1) < data[MACD_SIGNAL_LABEL].shift(1))
    data['Bearish_Crossover'] = (data[MACD_LABEL] < data[MACD_SIGNAL_LABEL]) & (data[MACD_LABEL].shift(1) > data[MACD_SIGNAL_LABEL].shift(1))
    
    # Define bull and bear signal pairs for traning
    bullish_index = None
    pairs = []
    # With these SMI limits we are getting quite accurate buy signals, but they are way too rare - model may underfit a little.
    # TODO: try to find better features and/or limits to avoid model underfitting. 
    for index, row in data.iterrows():
        if row['Bullish_Crossover']:
            if row[SMI_LABEL] > 30:
                bullish_index = index 
        elif row['Bearish_Crossover'] and bullish_index is not None:
            if row[SMI_LABEL] < -30:
                # Store sell signal with connected but signal and their data. Buy signal buy price is our feature.
                pairs.append((bullish_index, index, data.loc[bullish_index, 'Close'], data.loc[index, 'Close'], data.loc[index, SMI_LABEL]))
                bullish_index = None
            
    pair_df = pd.DataFrame(pairs, columns=['Bullish_Index', 'Bearish_Index', 'Bullish_Price', 'Bearish_Price', SMI_LABEL])
    # Define target variable (price at bearish crossover > price at bullish crossover)
    pair_df['Price_Increased'] = (pair_df['Bearish_Price'] > (pair_df['Bullish_Price'])).astype(int)
    price_increase_count = pair_df['Price_Increased'].sum()
    price_decreases_count = len(pair_df['Price_Increased']) - price_increase_count
    
    
    # Train Model
    X = pair_df[['Bullish_Price']] # feature(s)
    y = pair_df['Price_Increased']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Accuracy
    accuracy = model.score(X_test, y_test)
    
    probs = model.predict_proba(X_test)
    strong_signals = (probs[:, 1] > 0.6).astype(int)
    print(f"Model Accuracy: {accuracy:.2f}, price increase count: {price_increase_count}, strong signals: {sum(strong_signals)}, price decrease count: {price_decreases_count}")

if __name__ == "__main__":
    main()
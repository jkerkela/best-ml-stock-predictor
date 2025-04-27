import argparse
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

#NOTE: latest pandas_ta has problem with numpy versions >1.26.4, 
# see https://github.com/twopirllc/pandas-ta/issues/799 for reference

timestamp_column = "date"
TRAINING_DATA_FILE_PATH = "hf://datasets/gauss314/bitcoin_daily/data_btc.csv"
CURRENT_PRICE_SOURCE = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"

EPOCHS = 7
LAYERS = 3
NEURONS_PER_LAYER = 70
LEARNING_RATE= 0.000009
FEATURES = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
parser = argparse.ArgumentParser("stock_predict")
parser.add_argument('--retrain', dest='retrain', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = F.leaky_relu(out)
        return out

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length, 0]  # Predicting the 'Close' price
        sequences.append((seq, label))
    return sequences

def drop_NA_values(data_scaled):
    if np.isnan(data_scaled).any():
        print("Data contains NaN values")
        data_scaled = np.nan_to_num(data_scaled)
    return data_scaled

def train_model(model, train_sequences, val_sequences, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for seq, label in train_sequences:
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(DEVICE)  # Ensure label has shape [1, 1]
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            if torch.isnan(loss):
                print("NaN loss detected")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_sequences)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in val_sequences:
                seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                label = torch.tensor(label, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(DEVICE)
                output = model(seq)
                val_loss += criterion(output, label).item()
        
        val_loss /= len(val_sequences)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Initialize weights
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)

# Load the model and scaler
def load_model(model, model_path="lstm_model.pth", scaler_path="scaler.pkl"):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = joblib.load(scaler_path)
    return model, scaler

# Save the model and scaler
def save_model(model, scaler, epoch, model_path="lstm_model.pth", scaler_path="scaler.pkl"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, model_path)
    joblib.dump(scaler, scaler_path)

def get_bitcoin_prices(days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = CURRENT_PRICE_SOURCE
    params = {
        'vs_currency': 'usd',
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp())
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    prices = {datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d'): item[1] for item in data['prices']}
    return prices
    
def main():
    scaler = MinMaxScaler()
    model = LSTMModel(input_size=FEATURES, hidden_size=NEURONS_PER_LAYER, num_layers=LAYERS).to(DEVICE)

    if args.retrain:
        # Load and preprocess data
        data = pd.read_csv(
            TRAINING_DATA_FILE_PATH,
            parse_dates=[timestamp_column],  # Parse the timestamp values as dates.
        )
        # Rename the 'price' column to 'Close'
        data.rename(columns={'price': 'Close'}, inplace=True)
        # Calculate SMA indicators
        data['SMA_5'] = ta.sma(data['Close'], window=5)
        data['SMA_10'] = ta.sma(data['Close'], window=10)
        data = data.dropna()
        # Scale data
        data_scaled = scaler.fit_transform(data[['Close', 'SMA_5', 'SMA_10']])
        data_scaled = drop_NA_values(data_scaled)
        # Prepare training data
        seq_length = 10
        sequences = create_sequences(data_scaled, seq_length)
        train_size = int(len(sequences) * 0.8)
        train_sequences = sequences[:train_size]
        val_sequences = sequences[train_size:]
        model.apply(init_weights)
        # Training the model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(model, train_sequences, val_sequences, criterion, optimizer, num_epochs=EPOCHS)
        save_model(model, scaler, epoch=EPOCHS)

    # Query the model with current price and SMA values
    bitcoin_prices = get_bitcoin_prices(20)
    df = pd.DataFrame(list(bitcoin_prices.items()), columns=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)

    current_close = df.iloc[-1]['Close']
    print(f'Current price for asset is: {current_close}')

    # Infer the current data with model for price prediction
    if args.retrain:
        current_data_scaled = scaler.transform(df[['Close', 'SMA_5', 'SMA_10']])
    else:
        model, scaler = load_model(model)
        current_data_scaled = scaler.transform(df[['Close', 'SMA_5', 'SMA_10']])

    current_data_scaled[-1:]  # Select only the last row for prediction

    current_data_tensor = torch.tensor(current_data_scaled[-1:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        prediction = model(current_data_tensor)

    # Create a dummy array with the same shape as the original data
    dummy_data = np.zeros((1, FEATURES))
    dummy_data[0, 0] = prediction.item()  # Replace the 'Close' value with the predicted value
    # Inverse transform the dummy array
    predicted_movement = scaler.inverse_transform(dummy_data)
    print(f'Predicted stock price movement for asset is: {predicted_movement[0, 0]}')

if __name__ == "__main__":
    main()
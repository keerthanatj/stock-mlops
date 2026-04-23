import pandas as pd
import ta
import os

def prepare_data(filepath="data/INFY.csv"):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    # Rename columns to standard format if needed
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Keep only needed columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(inplace=True)

    # Feature engineering: technical indicators
    df['SMA_10']  = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_30']  = ta.trend.sma_indicator(df['Close'], window=30)
    df['RSI']     = ta.momentum.rsi(df['Close'], window=14)
    df['MACD']    = ta.trend.macd(df['Close'])
    df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_low']  = ta.volatility.bollinger_lband(df['Close'])

    # Target: 1 = price goes UP next day, 0 = goes DOWN
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/stock_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/stock_data.csv")
    print(df.head())

if __name__ == "__main__":
    prepare_data()
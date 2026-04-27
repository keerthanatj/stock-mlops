import pandas as pd
import ta
import os

df = pd.read_csv('data/INFY.csv')
df.columns = [c.strip() for c in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df[['Date','Open','High','Low','Close','Volume']].copy()
df.dropna(inplace=True)
df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
df['SMA_30'] = ta.trend.sma_indicator(df['Close'], window=30)
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['MACD'] = ta.trend.macd(df['Close'])
df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])
df['Return_1d'] = df['Close'].pct_change(1)
df['Return_5d'] = df['Close'].pct_change(5)
df['Volatility'] = df['Return_1d'].rolling(10).std()
df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
df['Close_Open'] = (df['Close'] - df['Open']) / df['Open']
df['Volume_MA10'] = df['Volume'].rolling(10).mean()
df['SMA_ratio'] = df['SMA_10'] / df['SMA_30']
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)
df.to_csv('data/stock_data.csv', index=False)
print('Total columns:', len(df.columns))
print('Columns:', df.columns.tolist())
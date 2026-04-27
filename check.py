import pandas as pd
df = pd.read_csv('data/stock_data.csv')
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(df['Target'].value_counts())
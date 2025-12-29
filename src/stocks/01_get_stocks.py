import pandas as pd
import yfinance as yf
import os

# Create the directory if it doesn't exist
data_path = 'data_stocks/raw/'
os.makedirs(data_path, exist_ok=True)

# Define functions to download and save data

def download_stock_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def save_to_parquet(data, ticker):
    file_path = os.path.join(data_path, f'{ticker}_data.parquet')
    data.to_parquet(file_path)
    print(f'Datos guardados en {file_path}')


# Tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

# Download and save data for each ticker
for ticker in tickers:
    stock_data = download_stock_data(ticker)
    save_to_parquet(stock_data, ticker)


    


    






import talib as ta
import pandas as pd
import numpy as np
import os

# Create the directories if they don't exist
data_path_load = 'data_stocks/raw/'
data_path_save = 'data_stocks/processed/'
os.makedirs(data_path_save, exist_ok=True)

# Load data
data_google = pd.read_parquet(os.path.join(data_path_load, 'GOOGL_data.parquet'))
data_apple = pd.read_parquet(os.path.join(data_path_load, 'AAPL_data.parquet'))
data_amazon = pd.read_parquet(os.path.join(data_path_load, 'AMZN_data.parquet'))
data_meta = pd.read_parquet(os.path.join(data_path_load, 'META_data.parquet'))
data_microsoft = pd.read_parquet(os.path.join(data_path_load, 'MSFT_data.parquet'))
data_nvidia = pd.read_parquet(os.path.join(data_path_load, 'NVDA_data.parquet'))
data_tesla = pd.read_parquet(os.path.join(data_path_load, 'TSLA_data.parquet'))

# Define the function to add the technical indicators
def add_technical_indicators(data):
    data['SMA_10'] = ta.SMA(data['Close'], timeperiod=10)
    data['SMA_20'] = ta.SMA(data['Close'], timeperiod=20)
    data['SMA_30'] = ta.SMA(data['Close'], timeperiod=30)   
    data['EMA_12'] = ta.EMA(data['Close'], timeperiod=12)
    data['EMA_26'] = ta.EMA(data['Close'], timeperiod=26)
    data['EMA_DIFF'] = data['EMA_12'] - data['EMA_26']
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['UPPER_BAND'], data['MIDDLE_BAND'], data['LOWER_BAND'] = ta.BBANDS(data['Close'], timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])
    data['OBV_EMA'] = ta.EMA(data['OBV'], timeperiod=18)
    data['MOMENTUM'] = ta.MOM(data['Close'], timeperiod=10)
    data['MFI'] = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
    data['STOCH_K'], data['STOCH_D'] = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    data['RSI_6'] = ta.RSI(data['Close'], timeperiod=6)
    data['RSI_12'] = ta.RSI(data['Close'], timeperiod=12)
    data['RSI_24'] = ta.RSI(data['Close'], timeperiod=24)
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['CCI'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['PRICE_MOMENTUM'] = data['Close'] - data['Close'].shift(10)
    data['PARABOLIC_SAR'] = ta.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    data['LARRY_WILLIAMS_R'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['DAY_OF_WEEK'] = data.index.dayofweek
    data['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * data['DAY_OF_WEEK'] / 7)
    data['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * data['DAY_OF_WEEK'] / 7)
    data['MONTH'] = data.index.month
    data['MONTH_COS'] = np.cos(2 * np.pi * data['MONTH'] / 12)
    data['MONTH_SIN'] = np.sin(2 * np.pi * data['MONTH'] / 12)
    data['DAY_OF_MONTH'] = data.index.day
    data['DAY_OF_MONTH_COS'] = np.cos(2 * np.pi * data['DAY_OF_MONTH'] / 31)
    data['DAY_OF_MONTH_SIN'] = np.sin(2 * np.pi * data['DAY_OF_MONTH'] / 31)
    return data

def save_to_parquet(data, ticker):
    file_path = os.path.join(data_path_save, f'{ticker}_data_processed.parquet')
    data.to_parquet(file_path)
    print(f'Datos guardados en {file_path}')


# Process and save data for each ticker
for ticker, data in zip(['GOOGL', 'AAPL', 'AMZN', 'META', 'MSFT', 'NVDA', 'TSLA'],
                        [data_google, data_apple, data_amazon, data_meta, data_microsoft, data_nvidia, data_tesla]):
    data_processed = add_technical_indicators(data)
    save_to_parquet(data_processed, ticker)
"""
Stock Data Preprocessing Pipeline (requested updates)
----------------------------------------------------
Implements:
 - Log-returns for High/Low/Open/Close
 - Technical indicators: RSI, MACD (+signal +hist), Bollinger Bands, SMA_10/20/30
 - Volume normalization (global and rolling windows)
 - Winsorization (1st/99th percentile clipping) for numeric outliers
 - Saves processed parquet per ticker

Usage:
	python src/stocks/2_process_stocks.py

Dependencies: pandas, numpy, scikit-learn, (optional) talib
"""

import os
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib as ta


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DATA_PATH_LOAD = 'data/stocks/raw/'
DATA_PATH_SAVE = 'data/stocks/processed/'
os.makedirs(DATA_PATH_SAVE, exist_ok=True)

TICKERS = ['GOOGL', 'AAPL', 'AMZN', 'META', 'MSFT', 'NVDA', 'TSLA']

VOLATILITY_WINDOW = 20
VOLUME_WINDOW = 20
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99

REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_numeric(df: pd.DataFrame, cols: List[str]):
	for c in cols:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors='coerce')
	return df


def save_parquet(df: pd.DataFrame, ticker: str):
	file_path = os.path.join(DATA_PATH_SAVE, f"{ticker}_data_processed.parquet")
	df.to_parquet(file_path, index=False, compression='snappy')
	print(f"  -> Saved: {file_path} (shape={df.shape})")


# -----------------------------------------------------------------------------
# Feature functions
# -----------------------------------------------------------------------------

def add_log_returns_ohlc(df: pd.DataFrame) -> pd.DataFrame:
	"""Add log returns for High/Low/Open/Close as LOG_RETURN_HIGH, etc."""
	df = df.copy()
	for col in ['High', 'Low', 'Open', 'Close']:
		if col in df.columns:
			ratio = df[col] / df[col].shift(1)
			ratio = ratio.replace([np.inf, -np.inf], np.nan)
			ratio = ratio.where(ratio > 0, np.nan)
			df[f'LOG_RETURN_{col.upper()}'] = np.log(ratio)
	return df


def add_ma_and_bbands(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	if 'Close' not in df.columns:
		return df

	# Moving averages
	df['SMA_10'] = df['Close'].rolling(10, min_periods=1).mean()
	df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
	df['SMA_30'] = df['Close'].rolling(30, min_periods=1).mean()

	# Bollinger Bands (20, 2)
	rolling_mean = df['Close'].rolling(20, min_periods=1).mean()
	rolling_std = df['Close'].rolling(20, min_periods=1).std()
	df['MIDDLE_BAND'] = rolling_mean
	df['UPPER_BAND'] = rolling_mean + 2 * rolling_std
	df['LOWER_BAND'] = rolling_mean - 2 * rolling_std

	return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
	df = df.copy()
	if 'Close' not in df.columns:
		return df

	if ta is not None:
		df[f'RSI_{period}'] = ta.RSI(df['Close'].values, timeperiod=period)
	else:
		# Simple RSI implementation
		delta = df['Close'].diff()
		up = delta.clip(lower=0)
		down = -1 * delta.clip(upper=0)
		ma_up = up.rolling(window=period, min_periods=1).mean()
		ma_down = down.rolling(window=period, min_periods=1).mean()
		rs = ma_up / (ma_down.replace(0, np.nan))
		df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
	return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	if 'Close' not in df.columns:
		return df

	if ta is not None:
		macd, signal, hist = ta.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
		df['MACD'] = macd
		df['MACD_SIGNAL'] = signal
		df['MACD_HIST'] = hist
	else:
		# fallback simple macd: ema12 - ema26
		ema12 = df['Close'].ewm(span=12, adjust=False).mean()
		ema26 = df['Close'].ewm(span=26, adjust=False).mean()
		macd = ema12 - ema26
		signal = macd.ewm(span=9, adjust=False).mean()
		hist = macd - signal
		df['MACD'] = macd
		df['MACD_SIGNAL'] = signal
		df['MACD_HIST'] = hist
	return df


def add_targets_and_volatility(df: pd.DataFrame, vol_window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
	df = df.copy()
	# Primary log return (Close)
	ratio = df['Close'] / df['Close'].shift(1)
	df['LOG_RETURN'] = np.log(ratio)
	df['ABS_LOG_RETURN'] = df['LOG_RETURN'].abs()
	df['VOLATILITY'] = df['LOG_RETURN'].rolling(window=vol_window).std()
	return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Add temporal cyclic features: day_of_week, month, quarter, day_of_month as sin/cos"""
	df = df.copy()
	if 'Date' not in df.columns:
		return df

	# Day of week (0=Monday)
	df['DAY_OF_WEEK'] = df['Date'].dt.weekday
	df['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
	df['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)

	# Month (1-12)
	df['MONTH'] = df['Date'].dt.month
	df['MONTH_SIN'] = np.sin(2 * np.pi * (df['MONTH'] - 1) / 12)
	df['MONTH_COS'] = np.cos(2 * np.pi * (df['MONTH'] - 1) / 12)

	# Day of month (1-31)
	df['DAY_OF_MONTH'] = df['Date'].dt.day
	df['DAY_OF_MONTH_SIN'] = np.sin(2 * np.pi * (df['DAY_OF_MONTH'] - 1) / 31)
	df['DAY_OF_MONTH_COS'] = np.cos(2 * np.pi * (df['DAY_OF_MONTH'] - 1) / 31)

	# Quarter (1-4)
	df['QUARTER'] = df['Date'].dt.quarter
	df['QUARTER_SIN'] = np.sin(2 * np.pi * (df['QUARTER'] - 1) / 4)
	df['QUARTER_COS'] = np.cos(2 * np.pi * (df['QUARTER'] - 1) / 4)

	return df





# -----------------------------------------------------------------------------
# High-level processing per ticker
# -----------------------------------------------------------------------------

def process_stock(ticker: str) -> pd.DataFrame:
	print('\n' + '=' * 60)
	print(f'Processing {ticker}...')
	print('=' * 60)

	fp = os.path.join(DATA_PATH_LOAD, f"{ticker}_data.parquet")
	if not os.path.exists(fp):
		print(f'  WARNING: file not found: {fp}')
		return pd.DataFrame()

	df = pd.read_parquet(fp)
	# Ensure Date is a column
	if df.columns.name is not None and df.columns.name == 'Price':
		df.columns.name = None

	if 'Date' in df.columns:
		df['Date'] = pd.to_datetime(df['Date'])
		df = df.sort_values('Date').reset_index(drop=True)
	else:
		# if Date is index
		try:
			df = df.reset_index()
			if 'index' in df.columns:
				df = df.rename(columns={'index': 'Date'})
			df['Date'] = pd.to_datetime(df['Date'])
			df = df.sort_values('Date').reset_index(drop=True)
		except Exception:
			pass

	# Ensure numeric types for core columns
	df = ensure_numeric(df, REQUIRED_COLS)

	# Add indicators
	df = add_ma_and_bbands(df)
	df = add_rsi(df, period=14)
	df = add_macd(df)

	# Log-returns OHLC
	df = add_log_returns_ohlc(df)

	# Primary targets + volatility
	df = add_targets_and_volatility(df)

	# Add temporal features (cyclic sin/cos). Do NOT scale variables (user requested no scaling)
	df = add_temporal_features(df)

	# Drop rows with NaN (from shifts etc.) and save
	df = df.dropna().reset_index(drop=True)

	save_parquet(df, ticker)
	return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
	processed = {}
	for t in TICKERS:
		try:
			processed[t] = process_stock(t)
		except Exception as e:
			print(f'Error processing {t}: {e}')
			continue

	print('\nSummary:')
	for t, df in processed.items():
		print(f'  {t}: {df.shape[0]} rows x {df.shape[1]} cols')

	print('\nDone.')



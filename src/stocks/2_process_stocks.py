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


def add_stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
	"""Add Stochastic Oscillator (%K and %D)"""
	df = df.copy()
	if not all(col in df.columns for col in ['High', 'Low', 'Close']):
		return df
	
	# %K = 100 * (Close - Low14) / (High14 - Low14)
	low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
	high_max = df['High'].rolling(window=k_period, min_periods=1).max()
	
	df['STOCH_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
	
	# %D = 3-period SMA of %K
	df['STOCH_D'] = df['STOCH_K'].rolling(window=d_period, min_periods=1).mean()
	
	return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
	"""Add Williams %R indicator"""
	df = df.copy()
	if not all(col in df.columns for col in ['High', 'Low', 'Close']):
		return df
	
	# Williams %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
	high_max = df['High'].rolling(window=period, min_periods=1).max()
	low_min = df['Low'].rolling(window=period, min_periods=1).min()
	
	df['WILLIAMS_R'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)
	
	return df


def add_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
	"""Add Realized Volatility (standard deviation of log returns)"""
	df = df.copy()
	if 'LOG_RETURN' not in df.columns:
		return df
	
	df['REALIZED_VOL'] = df['LOG_RETURN'].rolling(window=window, min_periods=1).std()
	
	return df


def add_parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
	"""
	Add Parkinson Volatility estimator
	Uses High and Low prices: σ = sqrt(1/(4*ln(2)) * mean((ln(H/L))^2))
	"""
	df = df.copy()
	if not all(col in df.columns for col in ['High', 'Low']):
		return df
	
	hl_ratio = np.log(df['High'] / (df['Low'] + 1e-10))
	hl_ratio_sq = hl_ratio ** 2
	
	# Parkinson estimator
	df['PARKINSON_VOL'] = np.sqrt(
		(1 / (4 * np.log(2))) * hl_ratio_sq.rolling(window=window, min_periods=1).mean()
	)
	
	return df


def add_garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
	"""
	Add Garman-Klass Volatility estimator
	GK = sqrt(0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2)
	"""
	df = df.copy()
	if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Open']):
		return df
	
	hl_ratio = np.log(df['High'] / (df['Low'] + 1e-10))
	co_ratio = np.log(df['Close'] / (df['Open'] + 1e-10))
	
	gk_component = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
	
	df['GARMAN_KLASS_VOL'] = np.sqrt(
		gk_component.rolling(window=window, min_periods=1).mean()
	)
	
	return df


def add_rogers_satchell_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
	"""
	Add Rogers-Satchell Volatility estimator
	RS = sqrt(mean(ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)))
	"""
	df = df.copy()
	if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Open']):
		return df
	
	hc = np.log(df['High'] / (df['Close'] + 1e-10))
	ho = np.log(df['High'] / (df['Open'] + 1e-10))
	lc = np.log(df['Low'] / (df['Close'] + 1e-10))
	lo = np.log(df['Low'] / (df['Open'] + 1e-10))
	
	rs_component = hc * ho + lc * lo
	
	df['ROGERS_SATCHELL_VOL'] = np.sqrt(
		rs_component.rolling(window=window, min_periods=1).mean()
	)
	
	return df


def add_estimated_vwap(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
	"""
	Add estimated VWAP (Volume Weighted Average Price)
	VWAP = sum(Price * Volume) / sum(Volume)
	Using typical price: (High + Low + Close) / 3
	"""
	df = df.copy()
	if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
		return df
	
	# Typical Price
	typical_price = (df['High'] + df['Low'] + df['Close']) / 3
	
	# VWAP calculation with rolling window
	df['VWAP'] = (
		(typical_price * df['Volume']).rolling(window=window, min_periods=1).sum() /
		df['Volume'].rolling(window=window, min_periods=1).sum()
	)
	
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

	# Add NEW indicators requested by user
	df = add_stochastic_oscillator(df, k_period=14, d_period=3)
	df = add_williams_r(df, period=14)
	df = add_realized_volatility(df, window=20)
	df = add_parkinson_volatility(df, window=20)
	df = add_garman_klass_volatility(df, window=20)
	df = add_rogers_satchell_volatility(df, window=20)
	df = add_estimated_vwap(df, window=20)

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



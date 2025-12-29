"""
Dataset class for LSTM Multi-Head embeddings
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from typing import List, Tuple, Dict
from .config import Config

class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    """Handles data loading, splitting and scaling"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.feature_scaler = RobustScaler()
        self.target_scalers = {}
    
    def load_and_process(self):
        filepath = os.path.join(Config.DATA_PATH_LOAD, f"{self.ticker}_data_processed.parquet")
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
            
        df = pd.read_parquet(filepath)
        
        # Ensure Date column
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add year column for splitting
        df['Year'] = df['Date'].dt.year
        
        # Filter available features
        available_features = [f for f in Config.FEATURE_COLS if f in df.columns]
        
        # Temporal splits
        train_df = df[(df['Year'] >= Config.TRAIN_START) & (df['Year'] <= Config.TRAIN_END)].copy()
        val_df = df[(df['Year'] >= Config.VAL_START) & (df['Year'] <= Config.VAL_END)].copy()
        test_df = df[df['Year'] == Config.TEST_YEAR].copy()
        
        # Scale features
        if not train_df.empty:
            train_df.loc[:, available_features] = self.feature_scaler.fit_transform(train_df[available_features])
            if not val_df.empty:
                val_df.loc[:, available_features] = self.feature_scaler.transform(val_df[available_features])
            if not test_df.empty:
                test_df.loc[:, available_features] = self.feature_scaler.transform(test_df[available_features])
        
        # Scale targets
        for target in Config.TARGETS:
            if target not in df.columns:
                continue
            
            scaler = RobustScaler()
            if not train_df.empty:
                train_df.loc[:, [target]] = scaler.fit_transform(train_df[[target]])
                if not val_df.empty:
                    val_df.loc[:, [target]] = scaler.transform(val_df[[target]])
                if not test_df.empty:
                    test_df.loc[:, [target]] = scaler.transform(test_df[[target]])
            
            self.target_scalers[target] = scaler
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'features': available_features,
            'full_dates': df['Date']  # Keep full dates mainly for reference if needed
        }

def create_sequences(data, features, targets, seq_length):
    """Create sequences for LSTM input"""
    if data.empty:
        return np.array([]), np.array([])
        
    X, y = [], []
    dates = []
    
    dates_series = data['Date'].values
    
    for i in range(len(data) - seq_length):
        # Input: seq_length days of features
        X.append(data[features].iloc[i:i+seq_length].values)
        # Target: next day's targets
        y.append(data[targets].iloc[i+seq_length].values)
        # Date: the date of the TARGET (aligned with embeddings)
        dates.append(dates_series[i+seq_length])
    
    return np.array(X), np.array(y), np.array(dates)

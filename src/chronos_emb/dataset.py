"""
Dataset class for Chronos embeddings
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from typing import List, Tuple

class StockWindowDataset(Dataset):
    """
    Dataset that creates sliding windows of stock time series.
    Compatible with Chronos input requirements.
    """
    def __init__(
        self, 
        tickers: List[str], 
        window_size: int, 
        target_col: str, 
        data_path: str,
        stride: int = 1
    ):
        self.window_size = window_size
        self.target_col = target_col
        self.stride = stride
        
        self.windows = []
        self.tickers_list = []
        self.dates_list = []
        
        print(f"\nLoading data for {len(tickers)} tickers...")
        
        for ticker in tickers:
            file_path = f"{data_path}{ticker}_data_processed.parquet"
            
            if not os.path.exists(file_path):
                print(f"  ✗ {ticker}: File not found, skipping")
                continue
            
            try:
                df = pd.read_parquet(file_path)
                df = df.sort_values('Date').reset_index(drop=True)
                
                if target_col not in df.columns:
                    print(f"  ✗ {ticker}: Column '{target_col}' not found, skipping")
                    continue
                
                # Extract series
                series = df[target_col].values
                dates = df['Date'].values
                
                # Create sliding windows
                num_windows = 0
                for i in range(0, len(series) - window_size + 1, stride):
                    window = series[i:i + window_size]
                    
                    # Chronos needs clean data, skip if NaN
                    if not np.isnan(window).any():
                        self.windows.append(window)
                        self.tickers_list.append(ticker)
                        # Store the date of the LAST day in the window (t)
                        # This aligns with the embedding representing the state at time t
                        self.dates_list.append(dates[i + window_size - 1])
                        num_windows += 1
                
                print(f"  ✓ {ticker}: {num_windows} windows created")
                
            except Exception as e:
                print(f"  ✗ {ticker}: Error loading data - {e}")
                continue
        
        # Convert to numpy array for efficiency
        self.windows = np.array(self.windows, dtype=np.float32)
        
        print(f"\nDataset created:")
        print(f"  - Total windows: {len(self.windows)}")
        print(f"  - Shape: {self.windows.shape}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)

    def get_metadata(self, idx):
        return {
            'ticker': self.tickers_list[idx],
            'date': self.dates_list[idx]
        }

"""
Dataset classes for PatchTST training
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from typing import List


class StockWindowDataset(Dataset):
    """
    Dataset that creates sliding windows of stock time series.
    Each window contains a fixed number of days of a target column (e.g., LOG_RETURN).
    """
    def __init__(
        self, 
        tickers: List[str],
        window_size: int,
        target_col: str,
        data_path: str,
        stride: int = 1
    ):
        """
        Args:
            tickers: List of stock tickers to load
            window_size: Size of each window (e.g., 20 days)
            target_col: Column name to extract (e.g., 'LOG_RETURN')
            data_path: Path to processed parquet files
            stride: Stride for sliding window (1 = maximum overlap)
        """
        self.window_size = window_size
        self.target_col = target_col
        self.stride = stride
        
        # Storage for all windows
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
                
                # Extract series and fill NaN
                series = df[target_col].ffill().bfill().values
                dates = df['Date'].values
                
                # Create sliding windows
                num_windows = 0
                for i in range(0, len(series) - window_size + 1, stride):
                    window = series[i:i + window_size]
                    
                    # Skip if any NaN
                    if not np.isnan(window).any():
                        self.windows.append(window)
                        self.tickers_list.append(ticker)
                        self.dates_list.append(dates[i + window_size - 1])
                        num_windows += 1
                
                print(f"  ✓ {ticker}: {num_windows} windows created")
                
            except Exception as e:
                print(f"  ✗ {ticker}: Error loading data - {e}")
                continue
        
        # Convert to numpy array
        self.windows = np.array(self.windows, dtype=np.float32)
        
        print(f"\nDataset created:")
        print(f"  - Total windows: {len(self.windows)}")
        print(f"  - Window size: {window_size} days")
        print(f"  - Stride: {stride}")
        print(f"  - Shape: {self.windows.shape}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Return a single window as a tensor."""
        return torch.tensor(self.windows[idx], dtype=torch.float32)
    
    def get_metadata(self, idx):
        """
        Get metadata for a specific window.
        
        Args:
            idx: Index of the window
            
        Returns:
            Dictionary with ticker and date
        """
        return {
            'ticker': self.tickers_list[idx],
            'date': self.dates_list[idx]
        }
    
    def get_statistics(self):
        """Get dataset statistics."""
        return {
            'num_windows': len(self.windows),
            'window_size': self.window_size,
            'mean': self.windows.mean(),
            'std': self.windows.std(),
            'min': self.windows.min(),
            'max': self.windows.max()
        }


def create_train_val_split(dataset, train_ratio=0.8, seed=42):
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: StockWindowDataset instance
        train_ratio: Ratio of data for training (default: 0.8)
        seed: Random seed for reproducibility
        
    Returns:
        train_subset, val_subset: Train and validation subsets
    """
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    print(f"\nDataset split:")
    print(f"  - Training: {len(train_subset)} windows ({train_ratio*100:.0f}%)")
    print(f"  - Validation: {len(val_subset)} windows ({(1-train_ratio)*100:.0f}%)")
    
    return train_subset, val_subset

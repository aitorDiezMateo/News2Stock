"""
Dataset classes for Stock Price Movement Prediction
====================================================
Handles loading, combining, and preprocessing stock and news embeddings.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
from dateutil import parser as date_parser

from .config import Config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stock_embeddings(ticker: str) -> pd.DataFrame:
    """
    Load stock embeddings for a specific ticker.
    
    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        
    Returns:
        DataFrame with Date, Ticker, and embedding columns
    """
    file_path = f"{Config.STOCK_EMBEDDINGS_PATH}{ticker}_embeddings.parquet"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock embeddings not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def load_news_embeddings(ticker: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load news embeddings for a specific ticker.
    
    Each ticker has its own news file containing ONLY news articles
    about that specific company.
    
    Mapping:
        AAPL -> apple_embeddings_*.parquet
        AMZN -> amazon_embeddings_*.parquet
        GOOGL -> google_embeddings_*.parquet
        META -> meta_embeddings_*.parquet
        MSFT -> microsoft_embeddings_*.parquet
        NVDA -> nvidia_embeddings_*.parquet
        TSLA -> tesla_embeddings_*.parquet
    
    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        verbose: Print detailed loading information
        
    Returns:
        DataFrame with Date and embedding columns
    """
    news_prefix = Config.TICKER_TO_NEWS.get(ticker)
    if news_prefix is None:
        raise ValueError(f"Unknown ticker: {ticker}")
    
    # Determine embedding type
    suffix = 'contextual' if Config.NEWS_EMBEDDING_TYPE == 'contextual' else 'no_context'
    file_path = f"{Config.NEWS_EMBEDDINGS_PATH}{news_prefix}_embeddings_{suffix}.parquet"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"News embeddings not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Parse dates from 'created' column
    def parse_date(date_str):
        try:
            return date_parser.parse(date_str).date()
        except:
            return None
    
    df['Date'] = df['created'].apply(parse_date)
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    if verbose:
        print(f"    📰 Loaded news for {ticker}:")
        print(f"       File: {os.path.basename(file_path)}")
        print(f"       Articles: {len(df)}")
        print(f"       Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df[['Date', 'embedding']]


def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load processed stock data (for volatility and returns).
    
    Args:
        ticker: Stock ticker
        
    Returns:
        DataFrame with price data and computed features
    """
    file_path = f"{Config.STOCK_DATA_PATH}{ticker}_data_processed.parquet"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock data not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def compute_target_label(
    current_price: float,
    next_price: float,
    volatility: float,
    threshold: float = 0.5
) -> int:
    """
    Compute target class based on price change and volatility.
    
    Args:
        current_price: Price at day 20
        next_price: Price at day 21
        volatility: Historical volatility
        threshold: Multiplier for neutral zone (default 0.5)
        
    Returns:
        0 (DOWN), 1 (NEUTRAL), or 2 (UP)
    """
    # Calculate return
    price_return = (next_price - current_price) / current_price
    
    # Define neutral zone as ±threshold * volatility
    neutral_zone = threshold * volatility
    
    if price_return < -neutral_zone:
        return 0  # DOWN
    elif price_return > neutral_zone:
        return 2  # UP
    else:
        return 1  # NEUTRAL


def get_window_news_embedding(
    news_df: pd.DataFrame,
    end_date: pd.Timestamp,
    window_days: int = None
) -> Optional[np.ndarray]:
    """
    Get mean embedding of news from the prediction window.
    
    Uses the same window size as the stock embeddings (default 20 days).
    
    Args:
        news_df: DataFrame with news embeddings (for a specific company)
        end_date: Last day of the window
        window_days: Number of days to look back (default: Config.WINDOW_SIZE = 20)
        
    Returns:
        Mean embedding vector or None if no news found
    """
    if window_days is None:
        window_days = Config.WINDOW_SIZE  # Use same window as stock embeddings (20 days)
    
    start_date = end_date - timedelta(days=window_days)
    
    # Filter news in the date range
    mask = (news_df['Date'] >= start_date) & (news_df['Date'] <= end_date)
    week_news = news_df[mask]
    
    if len(week_news) == 0:
        return None
    
    # Compute mean embedding
    embeddings = np.stack(week_news['embedding'].values)
    mean_embedding = embeddings.mean(axis=0)
    
    return mean_embedding


def create_combined_dataset(
    tickers: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    """
    Create combined dataset with stock and news embeddings.
    
    Args:
        tickers: List of tickers to process (default: all)
        verbose: Print progress information
        
    Returns:
        Tuple of (features, labels, ticker_indices, ticker_names, dates)
    """
    if tickers is None:
        tickers = Config.TICKERS
    
    all_features = []
    all_labels = []
    all_ticker_indices = []
    all_dates = []
    
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    for ticker in tickers:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print(f"{'='*50}")
        
        try:
            # Load data
            stock_emb_df = load_stock_embeddings(ticker)
            news_df = load_news_embeddings(ticker, verbose=verbose)  # Shows which file is loaded
            stock_data_df = load_stock_data(ticker)
            
            if verbose:
                print(f"    📈 Stock embeddings: {len(stock_emb_df)} windows")
                print(f"    💹 Stock data points: {len(stock_data_df)}")
            
            # Create date index for stock data
            stock_data_df = stock_data_df.set_index('Date')
            
            valid_samples = 0
            skipped_no_news = 0
            skipped_no_next_day = 0
            
            for idx, row in stock_emb_df.iterrows():
                window_end_date = row['Date']
                stock_embedding = np.array(row['embedding'])
                
                # Get next trading day data
                next_day_mask = stock_data_df.index > window_end_date
                if not next_day_mask.any():
                    skipped_no_next_day += 1
                    continue
                
                next_day_idx = stock_data_df.index[next_day_mask][0]
                
                # Get current and next day prices
                if window_end_date not in stock_data_df.index:
                    skipped_no_next_day += 1
                    continue
                    
                current_price = stock_data_df.loc[window_end_date, 'Close']
                next_price = stock_data_df.loc[next_day_idx, 'Close']
                volatility = stock_data_df.loc[window_end_date, 'VOLATILITY']
                
                # Get news embedding for the same 20-day window
                # news_df contains ONLY news for this specific company (e.g., Tesla news for TSLA)
                news_embedding = get_window_news_embedding(news_df, window_end_date)
                
                if news_embedding is None:
                    # Use zero vector if no news (could also skip)
                    news_embedding = np.zeros(Config.NEWS_EMBEDDING_DIM)
                    skipped_no_news += 1
                
                # Compute target label
                label = compute_target_label(
                    current_price, 
                    next_price, 
                    volatility,
                    Config.NEUTRAL_THRESHOLD
                )
                
                # Combine features
                if Config.INCLUDE_TICKER_FEATURE:
                    # One-hot encoding for ticker
                    ticker_onehot = np.zeros(len(tickers))
                    ticker_onehot[ticker_to_idx[ticker]] = 1.0
                    features = np.concatenate([stock_embedding, news_embedding, ticker_onehot])
                else:
                    features = np.concatenate([stock_embedding, news_embedding])
                
                all_features.append(features)
                all_labels.append(label)
                all_ticker_indices.append(ticker_to_idx[ticker])
                all_dates.append(window_end_date)
                valid_samples += 1
            
            if verbose:
                print(f"    ✅ Valid samples: {valid_samples}")
                if skipped_no_news > 0:
                    print(f"    ⚠️ Windows with no news (used zero vector): {skipped_no_news}")
                if skipped_no_next_day > 0:
                    print(f"    ⚠️ Skipped (no next day data): {skipped_no_next_day}")
                
        except FileNotFoundError as e:
            print(f"    ❌ Skipping {ticker}: {e}")
            continue
    
    # Convert to arrays
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    ticker_indices = np.array(all_ticker_indices, dtype=np.int64)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"  Total samples: {len(features)}")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"    - Stock embedding: {Config.STOCK_EMBEDDING_DIM}")
        print(f"    - News embedding: {Config.NEWS_EMBEDDING_DIM}")
        if Config.INCLUDE_TICKER_FEATURE:
            print(f"    - Ticker one-hot: {len(tickers)}")
        print(f"\n  Class distribution:")
        for i, name in enumerate(Config.CLASS_NAMES):
            count = (labels == i).sum()
            print(f"    {name}: {count} ({count/len(labels)*100:.1f}%)")
    
    return features, labels, ticker_indices, tickers, all_dates


class PredictionDataset(Dataset):
    """
    PyTorch Dataset for stock prediction.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        ticker_indices: Optional[np.ndarray] = None
    ):
        """
        Args:
            features: Input features [N, D]
            labels: Target labels [N]
            ticker_indices: Ticker indices for each sample [N]
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ticker_indices = ticker_indices
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        class_counts = torch.bincount(self.labels, minlength=Config.NUM_CLASSES)
        total = len(self.labels)
        weights = total / (Config.NUM_CLASSES * class_counts.float())
        return weights


def create_data_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    ticker_indices: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        features: Input features
        labels: Target labels
        ticker_indices: Ticker indices
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        batch_size: Batch size for DataLoaders
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    set_seed(seed)
    
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    
    # Split indices
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create datasets
    train_dataset = PredictionDataset(
        features[train_indices],
        labels[train_indices],
        ticker_indices[train_indices]
    )
    
    val_dataset = PredictionDataset(
        features[val_indices],
        labels[val_indices],
        ticker_indices[val_indices]
    )
    
    test_dataset = PredictionDataset(
        features[test_indices],
        labels[test_indices],
        ticker_indices[test_indices]
    )
    
    # Compute class weights from training set
    class_weights = train_dataset.get_class_weights()
    
    print(f"\nData Split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"\nClass weights: {class_weights.numpy()}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset creation...")
    features, labels, ticker_indices, tickers, dates = create_combined_dataset(verbose=True)
    
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        features, labels, ticker_indices,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        batch_size=Config.BATCH_SIZE,
        seed=Config.SEED
    )
    
    # Test batch
    X, y = next(iter(train_loader))
    print(f"\nBatch test:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y values: {y[:10]}")

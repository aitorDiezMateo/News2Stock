"""
Dataset for LSTM-based Stock Price Movement Prediction
=======================================================
Key difference: Aggregates news per DAY, creating a sequence of daily embeddings.
Output shape: (window_size, news_embedding_dim) - one embedding per day.
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
    """Load stock embeddings for a specific ticker."""
    file_path = f"{Config.STOCK_EMBEDDINGS_PATH}{ticker}_embeddings.parquet"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock embeddings not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def load_news_embeddings(ticker: str, verbose: bool = False) -> pd.DataFrame:
    """Load news embeddings for a specific ticker."""
    news_prefix = Config.TICKER_TO_NEWS.get(ticker)
    if news_prefix is None:
        raise ValueError(f"Unknown ticker: {ticker}")
    
    suffix = 'contextual' if Config.NEWS_EMBEDDING_TYPE == 'contextual' else 'no_context'
    file_path = f"{Config.NEWS_EMBEDDINGS_PATH}{news_prefix}_embeddings_{suffix}.parquet"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"News embeddings not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    
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
    """Load processed stock data (for volatility and returns)."""
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
    """Compute target class based on price change and volatility."""
    price_return = (next_price - current_price) / current_price
    neutral_zone = threshold * volatility
    
    if price_return < -neutral_zone:
        return 0  # DOWN
    elif price_return > neutral_zone:
        return 2  # UP
    else:
        return 1  # NEUTRAL


def get_daily_news_sequence(
    news_df: pd.DataFrame,
    end_date: pd.Timestamp,
    window_days: int = None,
    news_by_date: Dict = None
) -> np.ndarray:
    """
    Get sequence of daily aggregated news embeddings.
    
    KEY DIFFERENCE FROM ORIGINAL:
    - Instead of averaging ALL news in window, we average news WITHIN each day
    - Returns a sequence: (window_days, embedding_dim)
    
    Args:
        news_df: DataFrame with news embeddings (used if news_by_date not provided)
        end_date: Last day of the window
        window_days: Number of days to look back
        news_by_date: Pre-grouped news by date (for efficiency)
        
    Returns:
        Numpy array of shape (window_days, embedding_dim)
        Days with no news get zero vectors
    """
    if window_days is None:
        window_days = Config.WINDOW_SIZE
    
    embedding_dim = Config.NEWS_EMBEDDING_DIM
    daily_embeddings = np.zeros((window_days, embedding_dim), dtype=np.float32)
    
    # Calculate date range
    start_date = end_date - timedelta(days=window_days - 1)
    
    # For each day in the window
    for day_idx in range(window_days):
        current_date = start_date + timedelta(days=day_idx)
        current_date_key = current_date.date() if hasattr(current_date, 'date') else current_date
        
        # Get news for this specific day using pre-grouped dict (much faster)
        if news_by_date is not None:
            day_embeddings = news_by_date.get(current_date_key)
            if day_embeddings is not None and len(day_embeddings) > 0:
                daily_embeddings[day_idx] = np.mean(day_embeddings, axis=0)
        else:
            # Fallback to DataFrame filtering (slower)
            day_mask = news_df['Date'] == current_date_key
            day_news = news_df[day_mask]
            if len(day_news) > 0:
                embeddings = np.stack(day_news['embedding'].values)
                daily_embeddings[day_idx] = embeddings.mean(axis=0)
    
    return daily_embeddings


def pregroup_news_by_date(news_df: pd.DataFrame) -> Dict:
    """
    Pre-group news embeddings by date for efficient lookup.
    
    Returns:
        Dictionary mapping date (as datetime.date) -> numpy array of embeddings
    """
    news_by_date = {}
    # Ensure Date column is datetime
    news_df = news_df.copy()
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    for date, group in news_df.groupby(news_df['Date'].dt.date):
        news_by_date[date] = np.stack(group['embedding'].values)
    return news_by_date


def create_combined_dataset(
    tickers: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    """
    Create combined dataset with stock embeddings and daily news sequences.
    
    Returns:
        Tuple of (stock_embeddings, news_sequences, labels, ticker_indices, ticker_names, dates)
        - stock_embeddings: [N, stock_embedding_dim]
        - news_sequences: [N, window_size, news_embedding_dim]
        - labels: [N]
        - ticker_indices: [N]
    """
    if tickers is None:
        tickers = Config.TICKERS
    
    all_stock_embeddings = []
    all_news_sequences = []
    all_labels = []
    all_ticker_indices = []
    all_dates = []
    all_technical_features = []  # NEW: technical indicators
    
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    for ticker in tickers:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print(f"{'='*50}")
        
        try:
            stock_emb_df = load_stock_embeddings(ticker)
            news_df = load_news_embeddings(ticker, verbose=verbose)
            stock_data_df = load_stock_data(ticker)
            
            if verbose:
                print(f"    📈 Stock embeddings: {len(stock_emb_df)} windows")
                print(f"    💹 Stock data points: {len(stock_data_df)}")
            
            # Pre-group news by date for efficiency
            news_by_date = pregroup_news_by_date(news_df)
            if verbose:
                print(f"    📅 News dates with articles: {len(news_by_date)}")
            
            stock_data_df = stock_data_df.set_index('Date')
            
            valid_samples = 0
            skipped_no_next_day = 0
            
            for idx, row in stock_emb_df.iterrows():
                window_end_date = row['Date']
                stock_embedding = np.array(row['embedding'])
                
                # Get next trading day
                next_day_mask = stock_data_df.index > window_end_date
                if not next_day_mask.any():
                    skipped_no_next_day += 1
                    continue
                
                next_day_idx = stock_data_df.index[next_day_mask][0]
                
                if window_end_date not in stock_data_df.index:
                    skipped_no_next_day += 1
                    continue
                
                current_price = stock_data_df.loc[window_end_date, 'Close']
                next_price = stock_data_df.loc[next_day_idx, 'Close']
                volatility = stock_data_df.loc[window_end_date, 'VOLATILITY']
                
                # Get DAILY news sequence (key difference!) - using pre-grouped news
                news_sequence = get_daily_news_sequence(news_df, window_end_date, news_by_date=news_by_date)
                
                # Compute target label
                label = compute_target_label(
                    current_price, next_price, volatility, Config.NEUTRAL_THRESHOLD
                )
                
                # Extract PRICE FEATURES (same as benchmark)
                price_features = np.array([], dtype=np.float32)
                if Config.USE_PRICE_FEATURES:
                    price_feat_list = []
                    for feat_name in Config.PRICE_FEATURES:
                        if feat_name in stock_data_df.columns:
                            feat_value = stock_data_df.loc[window_end_date, feat_name]
                            if pd.isna(feat_value):
                                feat_value = 0.0
                            price_feat_list.append(feat_value)
                        else:
                            price_feat_list.append(0.0)
                    price_features = np.array(price_feat_list, dtype=np.float32)
                
                # Extract TECHNICAL FEATURES
                tech_features = np.array([], dtype=np.float32)
                if Config.USE_TECHNICAL_FEATURES:
                    tech_feat_list = []
                    for feat_name in Config.TECHNICAL_FEATURES:
                        if feat_name in stock_data_df.columns:
                            feat_value = stock_data_df.loc[window_end_date, feat_name]
                            # Handle NaN values
                            if pd.isna(feat_value):
                                feat_value = 0.0
                            tech_feat_list.append(feat_value)
                        else:
                            # Feature not found, use 0
                            tech_feat_list.append(0.0)
                    tech_features = np.array(tech_feat_list, dtype=np.float32)
                
                # Concatenate price + technical features
                additional_features = np.concatenate([price_features, tech_features])
                
                all_stock_embeddings.append(stock_embedding)
                all_news_sequences.append(news_sequence)
                all_technical_features.append(additional_features)
                all_labels.append(label)
                all_ticker_indices.append(ticker_to_idx[ticker])
                all_dates.append(window_end_date)
                valid_samples += 1
            
            if verbose:
                print(f"    ✅ Valid samples: {valid_samples}")
                if skipped_no_next_day > 0:
                    print(f"    ⚠️ Skipped (no next day data): {skipped_no_next_day}")
                
        except FileNotFoundError as e:
            print(f"    ❌ Skipping {ticker}: {e}")
            continue
    
    # Convert to arrays
    stock_embeddings = np.array(all_stock_embeddings, dtype=np.float32)
    news_sequences = np.array(all_news_sequences, dtype=np.float32)
    technical_features = np.array(all_technical_features, dtype=np.float32) if Config.USE_TECHNICAL_FEATURES else None
    labels = np.array(all_labels, dtype=np.int64)
    ticker_indices = np.array(all_ticker_indices, dtype=np.int64)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY (LSTM Strategy)")
        print(f"{'='*60}")
        print(f"  Total samples: {len(labels)}")
        print(f"  Stock embedding shape: {stock_embeddings.shape}")
        print(f"  News sequence shape: {news_sequences.shape}")
        print(f"    - {Config.WINDOW_SIZE} days x {Config.NEWS_EMBEDDING_DIM} dim")
        if Config.USE_TECHNICAL_FEATURES:
            print(f"  Technical features shape: {technical_features.shape}")
            print(f"    - {Config.TECHNICAL_FEATURES_DIM} indicators: {', '.join(Config.TECHNICAL_FEATURES[:5])}...")
        print(f"\n  Class distribution:")
        for i, name in enumerate(Config.CLASS_NAMES):
            count = (labels == i).sum()
            print(f"    {name}: {count} ({count/len(labels)*100:.1f}%)")
    
    return stock_embeddings, news_sequences, technical_features, labels, ticker_indices, tickers, all_dates


class LSTMPredictionDataset(Dataset):
    """
    PyTorch Dataset for LSTM-based prediction.
    
    Each sample contains:
    - stock_embedding: [stock_embedding_dim]
    - news_sequence: [window_size, news_embedding_dim]
    - technical_features: [technical_features_dim] (optional)
    - ticker_onehot: [num_tickers] (optional)
    - label: int
    """
    
    def __init__(
        self,
        stock_embeddings: np.ndarray,
        news_sequences: np.ndarray,
        technical_features: Optional[np.ndarray],
        labels: np.ndarray,
        ticker_indices: np.ndarray,
        include_ticker: bool = True
    ):
        self.stock_embeddings = torch.tensor(stock_embeddings, dtype=torch.float32)
        self.news_sequences = torch.tensor(news_sequences, dtype=torch.float32)
        self.technical_features = torch.tensor(technical_features, dtype=torch.float32) if technical_features is not None else None
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ticker_indices = ticker_indices
        self.include_ticker = include_ticker
        self.num_tickers = len(Config.TICKERS)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        stock_emb = self.stock_embeddings[idx]
        news_seq = self.news_sequences[idx]
        label = self.labels[idx]
        
        items = [stock_emb, news_seq]
        
        # Add technical features if available
        if self.technical_features is not None:
            tech_feat = self.technical_features[idx]
            items.append(tech_feat)
        
        # Add ticker onehot if enabled
        if self.include_ticker:
            ticker_onehot = torch.zeros(self.num_tickers)
            ticker_onehot[self.ticker_indices[idx]] = 1.0
            items.append(ticker_onehot)
        
        items.append(label)
        return tuple(items)


def create_data_loaders(
    stock_embeddings: np.ndarray,
    news_sequences: np.ndarray,
    technical_features: Optional[np.ndarray],
    labels: np.ndarray,
    ticker_indices: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test data loaders.
    """
    set_seed(seed)
    
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create datasets
    train_dataset = LSTMPredictionDataset(
        stock_embeddings[train_idx],
        news_sequences[train_idx],
        technical_features[train_idx] if technical_features is not None else None,
        labels[train_idx],
        ticker_indices[train_idx],
        include_ticker=Config.INCLUDE_TICKER_FEATURE
    )
    
    val_dataset = LSTMPredictionDataset(
        stock_embeddings[val_idx],
        news_sequences[val_idx],
        technical_features[val_idx] if technical_features is not None else None,
        labels[val_idx],
        ticker_indices[val_idx],
        include_ticker=Config.INCLUDE_TICKER_FEATURE
    )
    
    test_dataset = LSTMPredictionDataset(
        stock_embeddings[test_idx],
        news_sequences[test_idx],
        technical_features[test_idx] if technical_features is not None else None,
        labels[test_idx],
        ticker_indices[test_idx],
        include_ticker=Config.INCLUDE_TICKER_FEATURE
    )
    
    # Compute class weights
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=Config.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * Config.NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    print("Testing LSTM Dataset creation...")
    stock_emb, news_seq, labels, ticker_idx, tickers, dates = create_combined_dataset(verbose=True)
    print(f"\n✓ Dataset created successfully!")
    print(f"  Stock embeddings: {stock_emb.shape}")
    print(f"  News sequences: {news_seq.shape}")

"""
FinBERT-LSTM Experiment - Dataset
==================================
Prepares sequences for price prediction with sentiment features.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
from tqdm import tqdm

from .config import Config


def get_sentiment_analyzer():
    """Load FinBERT sentiment analyzer."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print("Loading FinBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.to(Config.DEVICE)
        model.eval()
        print("✓ FinBERT loaded")
        
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load FinBERT: {e}")
        return None, None


def compute_sentiment_scores(texts: List[str], tokenizer, model, batch_size: int = 64) -> np.ndarray:
    """
    Compute sentiment scores using FinBERT.
    
    Returns:
        Array of shape [n_texts, 3] with [positive, negative, neutral] scores
    """
    import torch
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_texts = [t[:512] if t and len(str(t).strip()) > 0 else "neutral" for t in batch_texts]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        
        # FinBERT outputs: [positive, negative, neutral]
        results.extend(probs)
    
    return np.array(results)


def load_stock_prices(ticker: str) -> pd.DataFrame:
    """Load stock price data."""
    file_path = os.path.join(Config.STOCK_DATA_PATH, f'{ticker}_data_processed.parquet')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock data not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]


def load_news_data(ticker: str) -> pd.DataFrame:
    """Load news data for a ticker."""
    company = Config.TICKER_TO_NEWS.get(ticker)
    file_path = os.path.join(Config.NEWS_EMBEDDINGS_PATH, f'{company}_embeddings_contextual.parquet')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"News data not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Parse dates
    if 'created' in df.columns:
        df['Date'] = pd.to_datetime(df['created']).dt.tz_localize(None)
    else:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    # Get text column
    text_col = 'text' if 'text' in df.columns else 'title'
    
    return df[['Date', text_col]].rename(columns={text_col: 'text'})


def load_stock_embeddings(ticker: str) -> pd.DataFrame:
    """Load stock embeddings from Chronos."""
    file_path = os.path.join(Config.STOCK_EMBEDDINGS_PATH, f'{ticker}_embeddings.parquet')
    
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df[['Date', 'embedding']]


def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    date: pd.Timestamp,
    tokenizer,
    model,
    window_days: int = 1
) -> np.ndarray:
    """
    Aggregate sentiment for news on a specific date (or window).
    
    Returns:
        Array of [positive, negative, neutral] mean scores
    """
    start_date = date - timedelta(days=window_days - 1)
    mask = (news_df['Date'].dt.date >= start_date.date()) & (news_df['Date'].dt.date <= date.date())
    day_news = news_df[mask]
    
    if len(day_news) == 0:
        return np.array([0.33, 0.33, 0.34])  # Neutral default
    
    texts = day_news['text'].fillna('').astype(str).tolist()
    sentiments = compute_sentiment_scores(texts, tokenizer, model)
    
    return sentiments.mean(axis=0)


def prepare_dataset_with_sentiment(
    ticker: str,
    tokenizer,
    model,
    use_stock_embeddings: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Prepare dataset for a single ticker with sentiment features.
    
    Returns:
        - sequences: [n_samples, seq_len, n_features]
        - targets: [n_samples] (next day's close price)
        - prices_for_scaling: [n_samples] (current close for % calculation)
        - dates: list of dates
    """
    if verbose:
        print(f"\nProcessing {ticker}...")
    
    # Load data
    price_df = load_stock_prices(ticker)
    news_df = load_news_data(ticker)
    
    stock_emb_df = None
    if use_stock_embeddings:
        stock_emb_df = load_stock_embeddings(ticker)
    
    if verbose:
        print(f"  Price data: {len(price_df)} days")
        print(f"  News data: {len(news_df)} articles")
    
    # Precompute daily sentiment for efficiency
    if verbose:
        print(f"  Computing daily sentiment...")
    
    unique_dates = price_df['Date'].unique()
    date_to_sentiment = {}
    
    for date in tqdm(unique_dates, desc="  Sentiment", disable=not verbose):
        sentiment = aggregate_daily_sentiment(news_df, date, tokenizer, model, window_days=1)
        date_to_sentiment[date] = sentiment
    
    # Create sequences
    sequences = []
    targets = []
    current_prices = []
    dates = []
    
    seq_len = Config.SEQUENCE_LENGTH
    horizon = Config.PREDICTION_HORIZON
    
    for i in range(seq_len, len(price_df) - horizon):
        # Get sequence dates and prices
        seq_dates = price_df['Date'].iloc[i-seq_len:i].values
        seq_prices = price_df['Close'].iloc[i-seq_len:i].values
        target_price = price_df['Close'].iloc[i + horizon - 1]
        current_price = price_df['Close'].iloc[i - 1]
        
        # Get sentiment for each day in sequence
        seq_sentiments = []
        for date in seq_dates:
            sent = date_to_sentiment.get(pd.Timestamp(date), np.array([0.33, 0.33, 0.34]))
            seq_sentiments.append(sent)
        seq_sentiments = np.array(seq_sentiments)
        
        # Normalize prices (min-max within sequence for stability)
        price_min, price_max = seq_prices.min(), seq_prices.max()
        if price_max > price_min:
            norm_prices = (seq_prices - price_min) / (price_max - price_min)
            # Normalize target using the SAME scaling
            norm_target = (target_price - price_min) / (price_max - price_min)
        else:
            norm_prices = np.ones_like(seq_prices) * 0.5
            norm_target = 0.5
        
        # Combine features: [sentiment(3), normalized_price(1)] = 4 features
        seq_features = np.column_stack([seq_sentiments, norm_prices])
        
        # Optionally add stock embeddings
        if use_stock_embeddings and stock_emb_df is not None:
            last_date = pd.Timestamp(seq_dates[-1])
            emb_mask = stock_emb_df['Date'] == last_date
            if emb_mask.any():
                stock_emb = np.array(stock_emb_df[emb_mask]['embedding'].values[0])
                # Repeat embedding for each timestep
                stock_emb_seq = np.tile(stock_emb, (seq_len, 1))
                seq_features = np.column_stack([seq_features, stock_emb_seq])
        
        sequences.append(seq_features)
        targets.append(norm_target)  # Store normalized target
        current_prices.append((target_price, current_price, price_min, price_max))  # Store actual prices + scaling
        dates.append(pd.Timestamp(price_df['Date'].iloc[i - 1]))
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    # current_prices now contains (target_price, current_price, price_min, price_max) tuples
    
    if verbose:
        print(f"  ✓ {len(sequences)} samples, shape: {sequences.shape}")
    
    return sequences, targets, current_prices, dates


def prepare_dataset_price_only(
    ticker: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Prepare dataset with only price data (for LSTM and DNN baselines).
    
    Returns:
        - sequences: [n_samples, seq_len, 1] for LSTM or [n_samples, seq_len] for DNN
        - targets: [n_samples]
        - prices_for_scaling: [n_samples]
        - dates: list of dates
    """
    if verbose:
        print(f"\nProcessing {ticker} (price only)...")
    
    price_df = load_stock_prices(ticker)
    
    sequences = []
    targets = []
    current_prices = []
    dates = []
    
    seq_len = Config.SEQUENCE_LENGTH
    horizon = Config.PREDICTION_HORIZON
    
    for i in range(seq_len, len(price_df) - horizon):
        seq_prices = price_df['Close'].iloc[i-seq_len:i].values
        target_price = price_df['Close'].iloc[i + horizon - 1]
        current_price = price_df['Close'].iloc[i - 1]
        
        # Normalize prices
        price_min, price_max = seq_prices.min(), seq_prices.max()
        if price_max > price_min:
            norm_prices = (seq_prices - price_min) / (price_max - price_min)
            # Normalize target using the SAME scaling
            norm_target = (target_price - price_min) / (price_max - price_min)
        else:
            norm_prices = np.ones_like(seq_prices) * 0.5
            norm_target = 0.5
        
        sequences.append(norm_prices)
        targets.append(norm_target)  # Store normalized target
        current_prices.append((target_price, current_price, price_min, price_max))  # Store actual prices + scaling
        dates.append(pd.Timestamp(price_df['Date'].iloc[i - 1]))
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    # current_prices now contains (target_price, current_price, price_min, price_max) tuples
    
    if verbose:
        print(f"  ✓ {len(sequences)} samples")
    
    return sequences, targets, current_prices, dates


class StockPriceDataset(Dataset):
    """Dataset for stock price prediction."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        scaling_info: list,  # List of (target_price, current_price, price_min, price_max) tuples
        for_lstm: bool = True
    ):
        """
        Args:
            sequences: Input sequences (normalized)
            targets: Target prices (normalized)
            scaling_info: List of tuples with (target_price, current_price, price_min, price_max)
            for_lstm: If True, ensure 3D shape [batch, seq, features]
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.scaling_info = scaling_info
        self.for_lstm = for_lstm
        
        # Ensure correct shape for LSTM
        if for_lstm and len(self.sequences.shape) == 2:
            self.sequences = self.sequences.unsqueeze(-1)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Return normalized target and scaling info as separate items
        target_price, current_price, price_min, price_max = self.scaling_info[idx]
        scaling = torch.tensor([target_price, current_price, price_min, price_max], dtype=torch.float32)
        return self.sequences[idx], self.targets[idx], scaling


def create_data_loaders(
    sequences: np.ndarray,
    targets: np.ndarray,
    current_prices: np.ndarray,
    for_lstm: bool = True,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders with temporal split.
    """
    n = len(sequences)
    train_end = int(n * Config.TRAIN_RATIO)
    val_end = int(n * (Config.TRAIN_RATIO + Config.VAL_RATIO))
    
    train_dataset = StockPriceDataset(
        sequences[:train_end], targets[:train_end], current_prices[:train_end], for_lstm
    )
    val_dataset = StockPriceDataset(
        sequences[train_end:val_end], targets[train_end:val_end], current_prices[train_end:val_end], for_lstm
    )
    test_dataset = StockPriceDataset(
        sequences[val_end:], targets[val_end:], current_prices[val_end:], for_lstm
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

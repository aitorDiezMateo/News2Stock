"""
Comprehensive Multimodal Experiment - Dataset
==============================================
Prepares data with all possible features for ablation studies.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .config import Config


def load_stock_prices(ticker: str) -> pd.DataFrame:
    """Load and compute indicators."""
    file_path = os.path.join(Config.STOCK_DATA_PATH, f'{ticker}_data_processed.parquet')
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Technical indicators
    df['returns'] = df['Close'].pct_change()
    df['volatility_5d'] = df['returns'].rolling(window=5).std()
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_ratio_5'] = df['Close'] / df['ma_5']
    df['ma_ratio_20'] = df['Close'] / df['ma_20']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs)) / 100
    
    # Volume
    df['volume_norm'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-10)
    
    # Future return for classification
    df['future_return'] = df['Close'].shift(-Config.PREDICTION_HORIZON) / df['Close'] - 1
    
    df = df.fillna(0)
    return df


def load_news_embeddings(ticker: str) -> pd.DataFrame:
    """Load news embeddings."""
    company = Config.TICKER_TO_NEWS.get(ticker)
    file_path = os.path.join(Config.NEWS_EMBEDDINGS_PATH, f'{company}_embeddings_contextual.parquet')
    
    df = pd.read_parquet(file_path)
    if 'created' in df.columns:
        df['Date'] = pd.to_datetime(df['created']).dt.tz_localize(None).dt.normalize()
    else:
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    
    return df[['Date', 'embedding', 'text']]


def load_timeseries_embeddings(ticker: str) -> Optional[pd.DataFrame]:
    """Load time series embeddings."""
    file_path = os.path.join(Config.TIMESERIES_EMBEDDINGS_PATH, f'{ticker}_embeddings.parquet')
    
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    return df[['Date', 'embedding']]


def compute_sentiment_finbert(texts: List[str]) -> np.ndarray:
    """Compute FinBERT sentiment scores."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.to(Config.DEVICE)
        model.eval()
        
        results = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t[:512] if t else "neutral" for t in batch_texts]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                             max_length=512, padding=True)
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            results.extend(probs)
        
        return np.array(results)
    except Exception as e:
        print(f"Warning: FinBERT failed: {e}")
        return np.array([[0.33, 0.33, 0.34]] * len(texts))


def compress_embedding(emb: np.ndarray, target_dim: int) -> np.ndarray:
    """Compress embedding."""
    if len(emb) <= target_dim:
        pad = np.zeros(target_dim - len(emb))
        return np.concatenate([emb, pad]).astype(np.float32)
    return emb[:target_dim].astype(np.float32)


def prepare_ticker_data(
    ticker: str,
    compute_sentiment: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """
    Prepare all features for a ticker.
    
    Returns list of samples with ALL possible features:
    - price_features: [seq_len, 7] (normalized price + 6 indicators)
    - news_embeddings: [seq_len, news_dim] (mean per day, compressed)
    - ts_embeddings: [seq_len, ts_dim] (compressed)
    - sentiment: [seq_len, 3] (FinBERT scores)
    - target_3class: 0=DOWN, 1=NEUTRAL, 2=UP
    - target_2class: 0=NEUTRAL, 1=CHANGE
    - future_return: actual return value
    """
    if verbose:
        print(f"\nProcessing {ticker}...")
    
    # Load data
    price_df = load_stock_prices(ticker)
    news_df = load_news_embeddings(ticker)
    ts_df = load_timeseries_embeddings(ticker)
    
    # Aggregate news by date
    news_by_date = {}
    for date, group in news_df.groupby(news_df['Date'].dt.date):
        embeddings = [emb for emb in group['embedding'].values]
        texts = list(group['text'].values)
        news_by_date[pd.Timestamp(date)] = {
            'embeddings': embeddings,
            'texts': texts
        }
    
    # TS embeddings by date
    ts_by_date = {}
    if ts_df is not None:
        for _, row in ts_df.iterrows():
            ts_by_date[row['Date']] = row['embedding']
    
    # Compute sentiment if requested
    if compute_sentiment and verbose:
        print(f"  Computing FinBERT sentiment...")
        for date, data in tqdm(list(news_by_date.items()), desc="  Sentiment"):
            if data['texts']:
                sentiments = compute_sentiment_finbert(data['texts'])
                data['sentiment'] = sentiments.mean(axis=0)
            else:
                data['sentiment'] = np.array([0.33, 0.33, 0.34])
    
    # Build samples
    samples = []
    seq_len = Config.SEQUENCE_LENGTH
    price_cols = ['returns', 'volatility_5d', 'ma_ratio_5', 'ma_ratio_20', 'rsi_14', 'volume_norm']
    
    for i in range(seq_len + 20, len(price_df) - Config.PREDICTION_HORIZON):
        future_return = price_df['future_return'].iloc[i-1]
        
        if pd.isna(future_return):
            continue
        
        # Classification targets
        if abs(future_return) <= Config.NEUTRAL_THRESHOLD:
            target_3class = 1  # NEUTRAL
            target_2class = 0  # NEUTRAL
        elif future_return > 0:
            target_3class = 2  # UP
            target_2class = 1  # CHANGE
        else:
            target_3class = 0  # DOWN
            target_2class = 1  # CHANGE
        
        # Get sequence
        seq_dates = price_df['Date'].iloc[i-seq_len:i].values
        seq_prices = price_df['Close'].iloc[i-seq_len:i].values
        
        # Normalize prices
        price_min, price_max = seq_prices.min(), seq_prices.max()
        if price_max > price_min:
            norm_prices = (seq_prices - price_min) / (price_max - price_min)
        else:
            norm_prices = np.ones_like(seq_prices) * 0.5
        
        # Build features for each timestep
        price_features = []
        news_embeddings = []
        ts_embeddings = []
        sentiments = []
        
        for j, date in enumerate(seq_dates):
            date_ts = pd.Timestamp(date).normalize()
            idx = i - seq_len + j
            
            # Price features: [normalized_close, returns, vol, ma_ratios, rsi, volume]
            pf = [norm_prices[j]]
            for col in price_cols:
                val = price_df[col].iloc[idx]
                if col in ['returns', 'volatility_5d']:
                    val = np.clip(val, -0.5, 0.5)
                elif col == 'volume_norm':
                    val = np.clip(val, -3, 3) / 3
                pf.append(val)
            price_features.append(pf)
            
            # News embedding (mean of day, compressed)
            if date_ts in news_by_date:
                day_news_embs = news_by_date[date_ts]['embeddings']
                if day_news_embs:
                    mean_emb = np.mean([emb for emb in day_news_embs], axis=0)
                    news_embeddings.append(compress_embedding(mean_emb, Config.NEWS_COMPRESSED_DIM))
                else:
                    news_embeddings.append(np.zeros(Config.NEWS_COMPRESSED_DIM, dtype=np.float32))
            else:
                news_embeddings.append(np.zeros(Config.NEWS_COMPRESSED_DIM, dtype=np.float32))
            
            # TS embedding
            if date_ts in ts_by_date:
                ts_emb = ts_by_date[date_ts]
                ts_embeddings.append(compress_embedding(np.array(ts_emb), Config.TS_COMPRESSED_DIM))
            else:
                ts_embeddings.append(np.zeros(Config.TS_COMPRESSED_DIM, dtype=np.float32))
            
            # Sentiment
            if compute_sentiment and date_ts in news_by_date:
                sentiments.append(news_by_date[date_ts].get('sentiment', np.array([0.33, 0.33, 0.34])))
            else:
                sentiments.append(np.array([0.33, 0.33, 0.34], dtype=np.float32))
        
        sample = {
            'price_features': np.array(price_features, dtype=np.float32),
            'news_embeddings': np.array(news_embeddings, dtype=np.float32),
            'ts_embeddings': np.array(ts_embeddings, dtype=np.float32),
            'sentiment': np.array(sentiments, dtype=np.float32),
            'target_3class': target_3class,
            'target_2class': target_2class,
            'future_return': future_return,
            'date': pd.Timestamp(price_df['Date'].iloc[i-1])
        }
        
        samples.append(sample)
    
    if verbose:
        print(f"  ✓ {len(samples)} samples")
    
    return samples


def prepare_all_tickers(compute_sentiment: bool = False, verbose: bool = True) -> Dict[str, List[Dict]]:
    """Prepare data for all tickers."""
    all_data = {}
    
    for ticker in Config.TICKERS:
        samples = prepare_ticker_data(ticker, compute_sentiment, verbose)
        all_data[ticker] = samples
    
    return all_data


class ClassificationDataset(Dataset):
    """Dataset for classification with feature selection."""
    
    def __init__(
        self,
        samples: List[Dict],
        task: str = '3class',  # '3class' or '2class'
        use_price: bool = True,
        use_news: bool = True,
        use_ts: bool = True,
        use_sentiment: bool = False
    ):
        self.samples = samples
        self.task = task
        self.use_price = use_price
        self.use_news = use_news
        self.use_ts = use_ts
        self.use_sentiment = use_sentiment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build feature vector
        features = []
        
        if self.use_price:
            features.append(sample['price_features'])
        if self.use_news:
            features.append(sample['news_embeddings'])
        if self.use_ts:
            features.append(sample['ts_embeddings'])
        if self.use_sentiment:
            features.append(sample['sentiment'])
        
        # Concatenate features
        if len(features) > 0:
            combined = np.concatenate(features, axis=-1)
        else:
            raise ValueError("No features selected!")
        
        # Get target
        target = sample[f'target_{self.task}']
        
        return {
            'features': torch.tensor(combined, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.long)
        }


def create_dataloaders(
    samples: List[Dict],
    task: str = '3class',
    use_price: bool = True,
    use_news: bool = True,
    use_ts: bool = True,
    use_sentiment: bool = False,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    
    n = len(samples)
    train_end = int(n * Config.TRAIN_RATIO)
    val_end = int(n * (Config.TRAIN_RATIO + Config.VAL_RATIO))
    
    train_ds = ClassificationDataset(samples[:train_end], task, use_price, use_news, use_ts, use_sentiment)
    val_ds = ClassificationDataset(samples[train_end:val_end], task, use_price, use_news, use_ts, use_sentiment)
    test_ds = ClassificationDataset(samples[val_end:], task, use_price, use_news, use_ts, use_sentiment)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

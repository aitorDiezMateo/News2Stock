"""
Comprehensive Multimodal Experiment - Configuration
===================================================
Tests all combinations of features and classification tasks.
"""
import os
import torch


class Config:
    """Configuration for comprehensive experiment."""
    
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    STOCK_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'stocks', 'processed')
    NEWS_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'news', 'embeddings')
    
    # Tickers
    TICKER_TO_NEWS = {
        'AAPL': 'apple',
        'AMZN': 'amazon',
        'GOOGL': 'google',
        'META': 'meta',
        'MSFT': 'microsoft',
        'NVDA': 'nvidia',
        'TSLA': 'tesla'
    }
    TICKERS = list(TICKER_TO_NEWS.keys())
    NUM_COMPANIES = len(TICKERS)
    
    # Sequence configuration
    SEQUENCE_LENGTH = 20
    PREDICTION_HORIZON = 1
    
    # Time series embedding configuration
    TIMESERIES_EMBEDDING_TYPE = 'chronos'  # 'chronos', 'lstm_multihead', or 'patchtst'
    TIMESERIES_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'embeddings', f'{TIMESERIES_EMBEDDING_TYPE}_{SEQUENCE_LENGTH}')
    
    RESULTS_PATH = os.path.join(ROOT_DIR, 'results', 'comprehensive_classification')
    
    # Classification thresholds
    NEUTRAL_THRESHOLD = 0.005  # ±0.5% = neutral
    
    # Embedding dimensions
    NEWS_EMBEDDING_DIM = 768
    TIMESERIES_EMBEDDING_DIM = 768
    SENTIMENT_DIM = 3  # FinBERT: [positive, negative, neutral]
    
    # Compressed dimensions
    NEWS_COMPRESSED_DIM = 64
    TS_COMPRESSED_DIM = 64
    
    # Model architecture
    LSTM_HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 15
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42

"""
FinBERT-LSTM Experiment Configuration
======================================
Based on: "Predicting Stock Prices with FinBERT-LSTM: 
          Integrating News Sentiment Analysis"

Three architectures compared:
1. FinBERT-LSTM: Sentiment (3 features) + Close prices (8 days)
2. LSTM: Only close prices (8 days)
3. DNN: Only close prices (8 days)

Plus additional variant:
4. FinBERT-LSTM-EMB: Sentiment + Close prices + Stock embeddings
"""
import os
import torch


class Config:
    """Configuration for FinBERT-LSTM experiment."""
    
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    STOCK_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'stocks', 'processed')
    NEWS_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'news', 'embeddings')
    STOCK_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'embeddings', 'chronos')
    RESULTS_PATH = os.path.join(ROOT_DIR, 'results', 'finbert_lstm')
    
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
    
    # Sequence configuration (from paper)
    SEQUENCE_LENGTH = 8  # 8 previous trading days
    PREDICTION_HORIZON = 1  # Predict next day's close price
    
    # Sentiment features
    SENTIMENT_DIM = 3  # positive, negative, neutral
    
    # Stock embeddings (Chronos)
    STOCK_EMBEDDING_DIM = 768
    USE_STOCK_EMBEDDINGS = False  # Set to True for enhanced variant
    
    # Model architecture (from paper)
    LSTM_HIDDEN_SIZE = 50
    LSTM_NUM_LAYERS = 3
    
    # DNN architecture (from paper)
    DNN_HIDDEN_DIMS = [256, 128, 64]
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Random seed
    SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create output directories."""
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)

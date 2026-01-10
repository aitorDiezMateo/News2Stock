"""
Configuration for LSTM-based Stock Price Movement Prediction
=============================================================
Uses daily news aggregation + LSTM/GRU for temporal sequence processing.
"""
import torch
import os


class Config:
    """Configuration class for LSTM-based stock prediction model."""
    
    # ========================================================================
    # PATHS
    # ========================================================================
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
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
    
    # Window configuration - CONFIGURABLE
    WINDOW_SIZE = 20  # Days in each window (configurable)
    
    # Stock embedding type: 'patchtst', 'chronos', 'lstm_multihead'
    STOCK_EMBEDDING_TYPE = 'lstm_multihead'
    
    # Stock embeddings path includes window size dynamically
    STOCK_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'embeddings', f'{STOCK_EMBEDDING_TYPE}_{WINDOW_SIZE}') + os.sep
    
    # Other data paths
    NEWS_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'news', 'embeddings') + os.sep
    STOCK_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'stocks', 'processed') + os.sep
    
    # Output paths (specific for LSTM approach)
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'prediction_lstm') + os.sep
    RESULTS_PATH = os.path.join(ROOT_DIR, 'results', 'prediction_lstm') + os.sep
    PLOTS_PATH = os.path.join(ROOT_DIR, 'plots', 'prediction_lstm') + os.sep
    
    # ========================================================================
    # PREDICTION HORIZON
    # ========================================================================
    PREDICTION_HORIZON = 1  # Days ahead to predict
    NEUTRAL_THRESHOLD = 0.5  # ±threshold * volatility = neutral zone
    
    # ========================================================================
    # PRICE FEATURES (same as benchmark)
    # ========================================================================
    # Base price features that benchmark models use
    USE_PRICE_FEATURES = False
    
    PRICE_FEATURES = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'VOLATILITY', 'RETURNS'
    ]
    
    PRICE_FEATURES_DIM = len(PRICE_FEATURES)
    
    # ========================================================================
    # TECHNICAL FEATURES
    # ========================================================================
    # Whether to include technical indicators as additional features
    USE_TECHNICAL_FEATURES = True
    
    # List of technical features to use (same as prediction module)
    TECHNICAL_FEATURES = [
        'RSI',           # Relative Strength Index (0-100)
        'MACD',          # Moving Average Convergence Divergence
        'MACD_SIGNAL',   # MACD Signal Line
        'MACD_HIST',     # MACD Histogram
        'BB_WIDTH',      # Bollinger Band Width (volatility)
        'MOMENTUM',      # Price momentum (10-day)
        'ROC',           # Rate of Change
        'VOLUME_RATIO',  # Volume vs 20-day average
        'ATR',           # Average True Range (normalized)
        'STOCH_K',       # Stochastic %K
        'STOCH_D',       # Stochastic %D
        'ADX',           # Average Directional Index (trend strength)
        'EMA_RATIO_12',  # Price / EMA(12) ratio
        'EMA_RATIO_26',  # Price / EMA(26) ratio
    ]
    
    # Total additional features (price + technical)
    ADDITIONAL_FEATURES_DIM = (PRICE_FEATURES_DIM if USE_PRICE_FEATURES else 0) + \
                             (len(TECHNICAL_FEATURES) if USE_TECHNICAL_FEATURES else 0)
    
    # Number of technical features
    TECHNICAL_FEATURES_DIM = len(TECHNICAL_FEATURES) if USE_TECHNICAL_FEATURES else 0
    
    # ========================================================================
    # EMBEDDING DIMENSIONS
    # ========================================================================
    STOCK_EMBEDDING_DIM = {
        'patchtst': 128,
        'chronos': 768,
        'lstm_multihead': 64
    }.get(STOCK_EMBEDDING_TYPE, 128)
    
    # News embedding type: 'contextual' or 'no_context'
    NEWS_EMBEDDING_TYPE = 'contextual'
    NEWS_EMBEDDING_DIM = 768 if NEWS_EMBEDDING_TYPE == 'contextual' else 300
    
    # ========================================================================
    # LSTM MODEL ARCHITECTURE
    # ========================================================================
    # LSTM for processing daily news sequence
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    LSTM_BIDIRECTIONAL = True
    
    # RNN type: 'lstm' or 'gru'
    RNN_TYPE = 'lstm'
    
    # Output dimension from LSTM (hidden_size * 2 if bidirectional)
    LSTM_OUTPUT_DIM = LSTM_HIDDEN_SIZE * 2 if LSTM_BIDIRECTIONAL else LSTM_HIDDEN_SIZE
    5 
    # ========================================================================
    # FUSION NETWORK (combines stock + news LSTM output)
    # ========================================================================
    NUM_CLASSES = 3  # DOWN, NEUTRAL, UP
    FUSION_HIDDEN_DIMS = [512, 256, 128]
    FUSION_DROPOUT = 0.3
    USE_BATCH_NORM = True
    ACTIVATION = 'leaky_relu'
    
    # Include ticker as feature
    INCLUDE_TICKER_FEATURE = True
    TICKER_EMBEDDING_DIM = len(TICKERS)
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    USE_SCHEDULER = True
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6
    
    EARLY_STOPPING_PATIENCE = 20
    USE_CLASS_WEIGHTS = True
    GRADIENT_CLIP = 1.0
    
    # ========================================================================
    # SYSTEM
    # ========================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    PIN_MEMORY = True if DEVICE == 'cuda' else False
    SEED = 42
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    CLASS_COLORS = ['#e74c3c', '#95a5a6', '#27ae60']
    
    @classmethod
    def get_input_dim(cls):
        """Get total input dimension for fusion network."""
        dim = cls.STOCK_EMBEDDING_DIM + cls.LSTM_OUTPUT_DIM
        if cls.USE_TECHNICAL_FEATURES:
            dim += cls.TECHNICAL_FEATURES_DIM
        if cls.INCLUDE_TICKER_FEATURE:
            dim += cls.TICKER_EMBEDDING_DIM
        return dim
    
    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("=" * 70)
        print("LSTM-BASED STOCK PREDICTION CONFIGURATION")
        print("=" * 70)
        
        print("\nData Configuration:")
        print(f"  - Tickers: {', '.join(cls.TICKERS)}")
        print(f"  - Window size: {cls.WINDOW_SIZE} days")
        print(f"  - Prediction horizon: {cls.PREDICTION_HORIZON} day(s)")
        print(f"  - Stock embedding type: {cls.STOCK_EMBEDDING_TYPE}")
        print(f"  - News embedding type: {cls.NEWS_EMBEDDING_TYPE}")
        
        print("\nEmbedding Dimensions:")
        print(f"  - Stock embedding: {cls.STOCK_EMBEDDING_DIM}")
        print(f"  - News embedding (per day): {cls.NEWS_EMBEDDING_DIM}")
        print(f"  - LSTM output: {cls.LSTM_OUTPUT_DIM}")
        
        print("\nLSTM Architecture:")
        print(f"  - RNN type: {cls.RNN_TYPE.upper()}")
        print(f"  - Hidden size: {cls.LSTM_HIDDEN_SIZE}")
        print(f"  - Num layers: {cls.LSTM_NUM_LAYERS}")
        print(f"  - Bidirectional: {cls.LSTM_BIDIRECTIONAL}")
        print(f"  - Dropout: {cls.LSTM_DROPOUT}")
        
        print("\nFusion Network:")
        print(f"  - Input dim: {cls.get_input_dim()}")
        print(f"  - Hidden dims: {cls.FUSION_HIDDEN_DIMS}")
        print(f"  - Output classes: {cls.NUM_CLASSES}")
        
        print("\nTraining:")
        print(f"  - Batch size: {cls.BATCH_SIZE}")
        print(f"  - Epochs: {cls.NUM_EPOCHS}")
        print(f"  - Learning rate: {cls.LEARNING_RATE}")
        print(f"  - Device: {cls.DEVICE}")
        
        print("=" * 70)
    
    @classmethod
    def create_directories(cls):
        """Create output directories."""
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        os.makedirs(cls.PLOTS_PATH, exist_ok=True)


if __name__ == "__main__":
    Config.print_config()

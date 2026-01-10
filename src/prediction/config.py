"""
Configuration for Stock Price Movement Prediction
==================================================
Contains all hyperparameters and settings for model training and evaluation.
"""
import torch
import os


class Config:
    """Configuration class for stock prediction model."""
    
    # ========================================================================
    # PATHS
    # ========================================================================
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    # Tickers to use (mapping stock ticker to news file prefix)
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
    
    # Window configuration
    WINDOW_SIZE = 5  # Days in each window
    
    # Stock embedding type: 'patchtst', 'chronos', 'lstm_multihead'
    STOCK_EMBEDDING_TYPE = 'chronos'
    
    # Stock embeddings path includes window size dynamically
    STOCK_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'embeddings', f'{STOCK_EMBEDDING_TYPE}_{WINDOW_SIZE}') + os.sep
    
    # Other data paths
    NEWS_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'data', 'news', 'embeddings') + os.sep
    STOCK_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'stocks', 'processed') + os.sep
    
    # Output paths
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'prediction') + os.sep
    RESULTS_PATH = os.path.join(ROOT_DIR, 'results', 'prediction') + os.sep
    PLOTS_PATH = os.path.join(ROOT_DIR, 'plots', 'prediction') + os.sep
    
    # ========================================================================
    # PREDICTION HORIZON
    # ========================================================================
    # How many days ahead to predict
    # 1 = predict next day (day 21 vs day 20)
    # 5 = predict 5 days ahead (day 25 vs day 20)
    # 10 = predict 10 days ahead (day 30 vs day 20)
    PREDICTION_HORIZON = 5  # Default: 5 days ahead
    
    # Neutral zone definition: ±NEUTRAL_THRESHOLD * volatility
    # For longer horizons, you may want to increase this
    NEUTRAL_THRESHOLD = 0.5
    
    # ========================================================================
    # TECHNICAL FEATURES
    # ========================================================================
    # Whether to include technical indicators as additional features
    USE_TECHNICAL_FEATURES = True
    
    # List of technical features to use
    # Available: RSI, MACD, MACD_SIGNAL, MACD_HIST, BB_UPPER, BB_LOWER, BB_WIDTH,
    #            MOMENTUM, ROC, VOLUME_RATIO, ATR, OBV_NORM, STOCH_K, STOCH_D,
    #            WILLIAMS_R, CCI, ADX, EMA_RATIO_12, EMA_RATIO_26
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
    
    # Number of technical features
    TECHNICAL_FEATURES_DIM = len(TECHNICAL_FEATURES) if USE_TECHNICAL_FEATURES else 0
    
    # News embedding type: 'contextual' or 'no_context'
    NEWS_EMBEDDING_TYPE = 'contextual'
    
    # ========================================================================
    # EMBEDDING DIMENSIONS
    # ========================================================================
    # Stock embedding dimension depends on type:
    # - 'patchtst': 128
    # - 'chronos': 768
    # - 'lstm_multihead': 256 (or configured value)
    STOCK_EMBEDDING_DIM = {
        'patchtst': 128,
        'chronos': 768,
        'lstm_multihead': 256
    }.get(STOCK_EMBEDDING_TYPE, 128)
    
    # News embedding dimension depends on type:
    # - 'no_context': 300 (GloVe/FastText)
    # - 'contextual': 768 (BERT/Transformer)
    NEWS_EMBEDDING_DIM = 768 if NEWS_EMBEDDING_TYPE == 'contextual' else 300
    
    # Combined input dimension (stock + news)
    INPUT_DIM = STOCK_EMBEDDING_DIM + NEWS_EMBEDDING_DIM
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    NUM_CLASSES = 3  # DOWN, NEUTRAL, UP
    
    # Hidden layer dimensions (4 layers)
    HIDDEN_DIMS = [512, 256, 128, 64]
    
    # Dropout probability
    DROPOUT = 0.3
    
    # Batch normalization
    USE_BATCH_NORM = True
    
    # Activation function: 'relu', 'leaky_relu', 'gelu', 'selu'
    ACTIVATION = 'leaky_relu'
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Training hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 regularization
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    
    # Class weighting for imbalanced data
    USE_CLASS_WEIGHTS = True
    
    # Gradient clipping
    GRADIENT_CLIP = 1.0
    
    # ========================================================================
    # TRAINING MODE
    # ========================================================================
    # 'unified': Single model for all tickers (recommended)
    # 'per_ticker': Separate model for each ticker
    TRAINING_MODE = 'unified'
    
    # Include ticker as feature (one-hot encoding)
    INCLUDE_TICKER_FEATURE = True
    TICKER_EMBEDDING_DIM = len(TICKERS)  # 7 for one-hot
    
    # Adjusted input dim if using ticker feature
    @classmethod
    def get_input_dim(cls):
        """Get total input dimension based on configuration."""
        dim = cls.STOCK_EMBEDDING_DIM + cls.NEWS_EMBEDDING_DIM
        if cls.INCLUDE_TICKER_FEATURE:
            dim += cls.TICKER_EMBEDDING_DIM
        if cls.USE_TECHNICAL_FEATURES:
            dim += cls.TECHNICAL_FEATURES_DIM
        return dim
    
    # ========================================================================
    # SYSTEM CONFIGURATION
    # ========================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    PIN_MEMORY = True if DEVICE == 'cuda' else False
    SEED = 42
    
    # ========================================================================
    # CLASS LABELS
    # ========================================================================
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    CLASS_COLORS = ['#e74c3c', '#95a5a6', '#27ae60']  # Red, Gray, Green
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("=" * 70)
        print("STOCK PREDICTION CONFIGURATION")
        print("=" * 70)
        
        print("\nData Configuration:")
        print(f"  - Tickers: {', '.join(cls.TICKERS)}")
        print(f"  - Window size: {cls.WINDOW_SIZE} days")
        print(f"  - Prediction horizon: {cls.PREDICTION_HORIZON} day(s)")
        print(f"  - Neutral threshold: ±{cls.NEUTRAL_THRESHOLD} * volatility")
        print(f"  - News embedding type: {cls.NEWS_EMBEDDING_TYPE}")
        
        print("\nEmbedding Dimensions:")
        print(f"  - Stock embedding: {cls.STOCK_EMBEDDING_DIM}")
        print(f"  - News embedding: {cls.NEWS_EMBEDDING_DIM}")
        print(f"  - Include ticker feature: {cls.INCLUDE_TICKER_FEATURE}")
        print(f"  - Technical features: {cls.USE_TECHNICAL_FEATURES} ({cls.TECHNICAL_FEATURES_DIM} features)")
        print(f"  - Total input dim: {cls.get_input_dim()}")
        
        print("\nModel Architecture:")
        print(f"  - Hidden layers: {cls.HIDDEN_DIMS}")
        print(f"  - Dropout: {cls.DROPOUT}")
        print(f"  - Batch normalization: {cls.USE_BATCH_NORM}")
        print(f"  - Activation: {cls.ACTIVATION}")
        print(f"  - Output classes: {cls.NUM_CLASSES}")
        
        print("\nTraining Configuration:")
        print(f"  - Training mode: {cls.TRAINING_MODE}")
        print(f"  - Batch size: {cls.BATCH_SIZE}")
        print(f"  - Epochs: {cls.NUM_EPOCHS}")
        print(f"  - Learning rate: {cls.LEARNING_RATE}")
        print(f"  - Weight decay: {cls.WEIGHT_DECAY}")
        print(f"  - Early stopping patience: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"  - Use class weights: {cls.USE_CLASS_WEIGHTS}")
        
        print("\nData Split:")
        print(f"  - Train: {cls.TRAIN_RATIO * 100:.0f}%")
        print(f"  - Validation: {cls.VAL_RATIO * 100:.0f}%")
        print(f"  - Test: {cls.TEST_RATIO * 100:.0f}%")
        
        print(f"\nSystem:")
        print(f"  - Device: {cls.DEVICE}")
        print(f"  - Seed: {cls.SEED}")
        
        print("=" * 70)
    
    @classmethod
    def create_directories(cls):
        """Create output directories if they don't exist."""
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        os.makedirs(cls.PLOTS_PATH, exist_ok=True)


if __name__ == "__main__":
    Config.print_config()

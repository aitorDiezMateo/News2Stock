"""
Configuration for Attention-based Stock Price Movement Prediction
==================================================================
Uses daily news aggregation + Self-Attention for temporal processing.
"""
import torch
import os


class Config:
    """Configuration class for Attention-based stock prediction model."""
    
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
    
    # Output paths (specific for Attention approach)
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'prediction_attention') + os.sep
    RESULTS_PATH = os.path.join(ROOT_DIR, 'results', 'prediction_attention') + os.sep
    PLOTS_PATH = os.path.join(ROOT_DIR, 'plots', 'prediction_attention') + os.sep
    
    # ========================================================================
    # PREDICTION HORIZON
    # ========================================================================
    PREDICTION_HORIZON = 1  # Days ahead to predict
    NEUTRAL_THRESHOLD = 0.5  # ±threshold * volatility = neutral zone
    
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
    # ATTENTION MODEL ARCHITECTURE
    # ========================================================================
    # Transformer/Attention for processing daily news sequence
    ATTENTION_DIM = 768  # Internal attention dimension (increased to preserve full resolution)
    NUM_ATTENTION_HEADS = 8
    NUM_TRANSFORMER_LAYERS = 2
    ATTENTION_DROPOUT = 0.1
    
    # Whether to use positional encoding
    USE_POSITIONAL_ENCODING = True
    
    # Attention output dimension (after aggregation)
    ATTENTION_OUTPUT_DIM = ATTENTION_DIM
    
    # ========================================================================
    # FUSION NETWORK (combines stock + news attention output)
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
    # Include technical indicators (RSI, MACD, Bollinger Bands, etc.)
    USE_TECHNICAL_FEATURES = True
    
    TECHNICAL_FEATURES = [
        'RSI',
        'MACD',
        'MACD_SIGNAL',
        'MACD_HIST',
        'BB_WIDTH',
        'MOMENTUM',
        'ROC',
        'VOLUME_RATIO',
        'ATR',
        'STOCH_K',
        'STOCH_D',
        'ADX',
        'EMA_RATIO_12',
        'EMA_RATIO_26'
    ]
    
    TECHNICAL_FEATURES_DIM = len(TECHNICAL_FEATURES)
    
    # Total additional features (price + technical)
    ADDITIONAL_FEATURES_DIM = (PRICE_FEATURES_DIM if USE_PRICE_FEATURES else 0) + \
                             (TECHNICAL_FEATURES_DIM if USE_TECHNICAL_FEATURES else 0)
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4  # Lower LR for transformers
    WEIGHT_DECAY = 1e-4
    
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'plateau'  # 'plateau' or 'cosine'
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6
    
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 1e-4
    USE_CLASS_WEIGHTS = True
    GRADIENT_CLIP = 1.0
    CHECKPOINT_FREQUENCY = 10
    
    # Output directories (aliases)
    MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'models', 'prediction_attention')
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'prediction_attention')
    PLOTS_DIR = os.path.join(ROOT_DIR, 'plots', 'prediction_attention')
    
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
        dim = cls.STOCK_EMBEDDING_DIM + cls.ATTENTION_OUTPUT_DIM
        if cls.USE_TECHNICAL_FEATURES:
            dim += cls.TECHNICAL_FEATURES_DIM
        if cls.INCLUDE_TICKER_FEATURE:
            dim += cls.TICKER_EMBEDDING_DIM
        return dim
    
    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("=" * 70)
        print("ATTENTION-BASED STOCK PREDICTION CONFIGURATION")
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
        print(f"  - Attention output: {cls.ATTENTION_OUTPUT_DIM}")
        if cls.USE_TECHNICAL_FEATURES:
            print(f"  - Technical features: {cls.TECHNICAL_FEATURES_DIM}")
        
        print("\nAttention Architecture:")
        print(f"  - Attention dim: {cls.ATTENTION_DIM}")
        print(f"  - Num attention heads: {cls.NUM_ATTENTION_HEADS}")
        print(f"  - Num transformer layers: {cls.NUM_TRANSFORMER_LAYERS}")
        print(f"  - Positional encoding: {cls.USE_POSITIONAL_ENCODING}")
        print(f"  - Dropout: {cls.ATTENTION_DROPOUT}")
        
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

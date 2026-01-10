"""
Configuration for Multi-Head LSTM Embeddings
"""
import torch
import os

class Config:
    """Configuration class for LSTM embedding extraction."""
    
    # ========================================================================
    # PATHS
    # ========================================================================
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH_LOAD = os.path.join(ROOT_DIR, 'data', 'stocks', 'processed') + os.sep
    RESULTS_PATH = os.path.join(ROOT_DIR, 'results', 'lstm') + os.sep
    
    # ========================================================================
    # DATA
    # ========================================================================
    TICKERS = ['GOOGL', 'AAPL', 'AMZN', 'META', 'MSFT', 'NVDA', 'TSLA']
    SEQUENCE_LENGTH = 20
    
    # EMBEDDINGS_SAVE_PATH includes sequence length (window size) dynamically
    EMBEDDINGS_SAVE_PATH = os.path.join(ROOT_DIR, 'data', 'embeddings', f'lstm_multihead_{SEQUENCE_LENGTH}') + os.sep
    
    TARGETS = ['LOG_RETURN', 'ABS_LOG_RETURN', 'VOLATILITY']
    TARGET_WEIGHTS = {
        'LOG_RETURN': 3.0,
        'ABS_LOG_RETURN': 2.0,
        'VOLATILITY': 1.0
    }
    
    # Feature columns (as defined in original script)
    FEATURE_COLS = [
        # Price data
        'Close', 'High', 'Low', 'Open', 'Volume',
        # Moving averages
        'SMA_10', 'SMA_20', 'SMA_30',
        # Bollinger Bands
        'UPPER_BAND', 'MIDDLE_BAND', 'LOWER_BAND',
        # MACD
        'MACD', 'MACD_SIGNAL', 'MACD_HIST',
        # RSI
        'RSI_14',
        # Stochastic Oscillator
        'STOCH_K', 'STOCH_D',
        # Williams %R
        'WILLIAMS_R',
        # Log returns
        'LOG_RETURN_HIGH', 'LOG_RETURN_LOW', 'LOG_RETURN_OPEN', 'LOG_RETURN_CLOSE',
        # Volatility estimators
        'REALIZED_VOL', 'PARKINSON_VOL', 'GARMAN_KLASS_VOL', 'ROGERS_SATCHELL_VOL',
        # VWAP
        'VWAP',
        # Temporal features (cyclic encoded)
        'DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS',
        'MONTH_SIN', 'MONTH_COS',
        'DAY_OF_MONTH_SIN', 'DAY_OF_MONTH_COS',
        'QUARTER_SIN', 'QUARTER_COS'
    ]
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    EPOCHS = 200
    PATIENCE = 25
    L2_REG = 1e-3
    
    # Temporal splits
    TRAIN_START = 2015
    TRAIN_END = 2021
    VAL_START = 2021
    VAL_END = 2023
    TEST_YEAR = 2024
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("="*70)
        print("LSTM MULTI-HEAD CONFIGURATION")
        print("="*70)
        print("\nData:")
        print(f"  - Tickers: {', '.join(cls.TICKERS)}")
        print(f"  - Sequence length: {cls.SEQUENCE_LENGTH}")
        
        print("\nModel:")
        print(f"  - Hidden Size: {cls.HIDDEN_SIZE}")
        print(f"  - Num Layers: {cls.NUM_LAYERS}")
        print(f"  - Dropout: {cls.DROPOUT}")
        print(f"  - Device: {cls.DEVICE}")
        
        print("\nOutput:")
        print(f"  - Embeddings path: {cls.EMBEDDINGS_SAVE_PATH}")
        print("="*70)

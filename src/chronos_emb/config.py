"""
Configuration for Chronos Embeddings Extraction
"""
import torch
import os

class Config:
    """Configuration class for Chronos embedding extraction."""
    
    # ========================================================================
    # PATHS
    # ========================================================================
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH_LOAD = os.path.join(ROOT_DIR, 'data', 'stocks', 'processed') + os.sep
    
    # ========================================================================
    # DATA
    # ========================================================================
    TICKERS = ['GOOGL', 'AAPL', 'AMZN', 'META', 'MSFT', 'NVDA', 'TSLA']
    TARGET_COL = 'LOG_RETURN'
    WINDOW_SIZE = 20
    STRIDE = 1
    
    # EMBEDDINGS_SAVE_PATH includes window size dynamically
    EMBEDDINGS_SAVE_PATH = os.path.join(ROOT_DIR, 'data', 'embeddings', f'chronos_{WINDOW_SIZE}') + os.sep
    
    # ========================================================================
    # MODEL
    # ========================================================================
    MODEL_NAME = "amazon/chronos-t5-base"
    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("="*70)
        print("CHRONOS CONFIGURATION")
        print("="*70)
        print("\nData:")
        print(f"  - Tickers: {', '.join(cls.TICKERS)}")
        print(f"  - Window size: {cls.WINDOW_SIZE} days")
        print(f"  - Target column: {cls.TARGET_COL}")
        print(f"  - Stride: {cls.STRIDE}")
        
        print("\nModel:")
        print(f"  - Name: {cls.MODEL_NAME}")
        print(f"  - Device: {cls.DEVICE}")
        
        print("\nProcessing:")
        print(f"  - Batch size: {cls.BATCH_SIZE}")
        print(f"  - Output path: {cls.EMBEDDINGS_SAVE_PATH}")
        print("="*70)

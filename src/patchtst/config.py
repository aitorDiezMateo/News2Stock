"""
Configuration for PatchTST MLM Training
"""
import torch


class Config:
    """Configuration class for PatchTST model and training."""
    
    # ========================================================================
    # DATA
    # ========================================================================
    TICKERS = ['GOOGL', 'AAPL', 'AMZN', 'META', 'MSFT', 'NVDA', 'TSLA']
    TARGET_COL = 'LOG_RETURN'  # Column to use for training
    WINDOW_SIZE = 20  # 20 trading days
    STRIDE = 1  # Stride for creating windows (1 = maximum overlap)
    
    # ========================================================================
    # PATHS (defined after WINDOW_SIZE to use it dynamically)
    # ========================================================================
    DATA_PATH_LOAD = 'data/stocks/processed/'
    MODEL_SAVE_PATH = 'models/patchtst/'
    EMBEDDINGS_SAVE_PATH = f'data/embeddings/patchtst_{WINDOW_SIZE}/'
    PLOTS_PATH = 'plots/patchtst/'
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    PATCH_SIZE = 4  # Each patch is 4 days
    PATCH_STRIDE = 2  # Overlapping patches with stride 2
    D_MODEL = 128  # Transformer dimension
    NHEAD = 8  # Number of attention heads
    NUM_LAYERS = 3  # Number of transformer layers
    DIM_FEEDFORWARD = 256  # Feedforward dimension
    DROPOUT = 0.1  # Dropout probability
    MASK_RATIO = 0.4  # Ratio of patches to mask (40%)
    MASK_TYPE = 'learnable'  # 'zero' or 'learnable'
    USE_REVIN = True  # Use Reversible Instance Normalization
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS = 5
    GRADIENT_CLIP = 1.0
    
    # Train/Val split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    
    # ========================================================================
    # SYSTEM
    # ========================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0  # DataLoader workers
    PIN_MEMORY = True if DEVICE == 'cuda' else False
    SEED = 42
    
    # ========================================================================
    # EMBEDDING EXTRACTION
    # ========================================================================
    POOLING_STRATEGY = 'mean'  # 'mean', 'max', 'last', or 'cls'
    EMBEDDING_BATCH_SIZE = 128
    
    @classmethod
    def get_num_patches(cls):
        """Calculate number of patches given window size and patch configuration."""
        return (cls.WINDOW_SIZE - cls.PATCH_SIZE) // cls.PATCH_STRIDE + 1
    
    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("="*70)
        print("CONFIGURATION")
        print("="*70)
        print("\nData:")
        print(f"  - Tickers: {', '.join(cls.TICKERS)}")
        print(f"  - Window size: {cls.WINDOW_SIZE} days")
        print(f"  - Target column: {cls.TARGET_COL}")
        print(f"  - Window stride: {cls.STRIDE}")
        
        print("\nModel Architecture:")
        print(f"  - Patch size: {cls.PATCH_SIZE}")
        print(f"  - Patch stride: {cls.PATCH_STRIDE}")
        print(f"  - Number of patches: {cls.get_num_patches()}")
        print(f"  - Model dimension: {cls.D_MODEL}")
        print(f"  - Attention heads: {cls.NHEAD}")
        print(f"  - Transformer layers: {cls.NUM_LAYERS}")
        print(f"  - Mask ratio: {cls.MASK_RATIO}")
        print(f"  - Use RevIN: {cls.USE_REVIN}")
        
        print("\nTraining:")
        print(f"  - Batch size: {cls.BATCH_SIZE}")
        print(f"  - Epochs: {cls.NUM_EPOCHS}")
        print(f"  - Learning rate: {cls.LEARNING_RATE}")
        print(f"  - Warmup epochs: {cls.WARMUP_EPOCHS}")
        print(f"  - Device: {cls.DEVICE}")
        
        print("="*70)

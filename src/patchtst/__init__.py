"""
PatchTST Package
================
Modular implementation of PatchTST with Masked Language Modeling
for time series embedding extraction.

Modules:
    - config: Configuration and hyperparameters
    - model: PatchTST model architecture
    - dataset: Dataset classes for time series windows
    - trainer: Training utilities
    - embeddings: Embedding extraction and saving
    - visualization: Plotting utilities
"""

from .config import Config
from .model import PatchTST_MLM, RevIN, Patching, PositionalEncoding, PatchMasking
from .dataset import StockWindowDataset, create_train_val_split
from .trainer import (
    set_seed, 
    get_lr_scheduler, 
    train_epoch, 
    validate,
    save_checkpoint,
    load_checkpoint
)
from .embeddings import (
    extract_embeddings,
    save_embeddings_by_ticker,
    load_embeddings
)
from .visualization import (
    plot_training_history,
    plot_learning_rate,
    plot_loss_distribution,
    plot_reconstruction_examples
)

__version__ = '1.0.0'

__all__ = [
    # Config
    'Config',
    
    # Model
    'PatchTST_MLM',
    'RevIN',
    'Patching',
    'PositionalEncoding',
    'PatchMasking',
    
    # Dataset
    'StockWindowDataset',
    'create_train_val_split',
    
    # Trainer
    'set_seed',
    'get_lr_scheduler',
    'train_epoch',
    'validate',
    'save_checkpoint',
    'load_checkpoint',
    
    # Embeddings
    'extract_embeddings',
    'save_embeddings_by_ticker',
    'load_embeddings',
    
    # Visualization
    'plot_training_history',
    'plot_learning_rate',
    'plot_loss_distribution',
    'plot_reconstruction_examples',
]

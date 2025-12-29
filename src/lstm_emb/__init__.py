"""
LSTM Multi-Head Embeddings Package
==================================
Modular implementation for training and extracting embeddings using Multi-Task LSTM.
"""

from .config import Config
from .dataset import DataProcessor, StockDataset
from .model import StockLSTMMultiHead
from .embeddings import extract_embeddings_from_loader, save_embeddings

__version__ = '1.0.0'

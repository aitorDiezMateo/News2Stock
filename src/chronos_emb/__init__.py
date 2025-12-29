"""
Chronos Embeddings Package
==========================
Modular implementation for extracting embeddings using Chronos-T5.
"""

from .config import Config
from .dataset import StockWindowDataset
from .embeddings import load_chronos_model, extract_embeddings, save_embeddings

__version__ = '1.0.0'

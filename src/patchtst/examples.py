"""
Example: Using PatchTST for Embedding Extraction
=================================================
Demonstrates how to use the trained PatchTST model to extract embeddings.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from patchtst import (
    Config,
    PatchTST_MLM,
    load_checkpoint,
    load_embeddings,
    StockWindowDataset
)


def example_1_load_pretrained_model():
    """Example 1: Load a pre-trained model and extract embeddings."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Load Pre-trained Model")
    print("="*70)
    
    # Create model with same architecture as training
    model = PatchTST_MLM(
        seq_len=Config.WINDOW_SIZE,
        patch_size=Config.PATCH_SIZE,
        stride=Config.PATCH_STRIDE,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        mask_ratio=Config.MASK_RATIO,
        mask_type=Config.MASK_TYPE,
        use_revin=Config.USE_REVIN
    )
    
    # Load checkpoint
    checkpoint_path = f"{Config.MODEL_SAVE_PATH}best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ Model not found: {checkpoint_path}")
        print("  Please train the model first using: python train.py")
        return None
    
    checkpoint = load_checkpoint(checkpoint_path, model)
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  - Epoch: {checkpoint['epoch'] + 1}")
    print(f"  - Train loss: {checkpoint['train_loss']:.6f}")
    print(f"  - Val loss: {checkpoint['val_loss']:.6f}")
    
    model.eval()
    return model


def example_2_extract_embeddings_from_data(model):
    """Example 2: Extract embeddings from new data."""
    if model is None:
        return
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Extract Embeddings from New Data")
    print("="*70)
    
    # Create synthetic data (20-day window)
    print("\nCreating synthetic 20-day window...")
    synthetic_data = torch.randn(1, Config.WINDOW_SIZE).cumsum(dim=1)
    
    print(f"  - Input shape: {synthetic_data.shape}")
    
    # Extract embeddings with different pooling strategies
    print("\nExtracting embeddings with different pooling:")
    
    pooling_strategies = ['mean', 'max', 'last', 'cls']
    
    for pooling in pooling_strategies:
        embedding = model.extract_embeddings(synthetic_data, pooling=pooling)
        print(f"  - {pooling:6s} pooling: {embedding.shape} | mean={embedding.mean():.4f}, std={embedding.std():.4f}")


def example_3_load_saved_embeddings():
    """Example 3: Load pre-computed embeddings."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Load Pre-computed Embeddings")
    print("="*70)
    
    # Try to load embeddings for each ticker
    for ticker in Config.TICKERS[:3]:  # Just first 3 for demo
        try:
            data = load_embeddings(ticker, Config.EMBEDDINGS_SAVE_PATH)
            
            print(f"\n✓ {ticker}:")
            print(f"  - Embeddings shape: {data['embeddings'].shape}")
            print(f"  - Date range: {data['dates'][0]} to {data['dates'][-1]}")
            print(f"  - Window size: {data['window_size']}")
            print(f"  - Target column: {data['target_col']}")
            
            # Show statistics
            emb = data['embeddings']
            print(f"  - Embedding stats: mean={emb.mean():.4f}, std={emb.std():.4f}")
            
        except FileNotFoundError:
            print(f"\n✗ {ticker}: Embeddings not found")
            print(f"  Please run training first: python train.py")


def example_4_batch_processing(model):
    """Example 4: Process multiple windows in batch."""
    if model is None:
        return
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    # Create batch of synthetic windows
    batch_size = 16
    synthetic_batch = torch.randn(batch_size, Config.WINDOW_SIZE).cumsum(dim=1)
    
    print(f"\nProcessing batch of {batch_size} windows...")
    print(f"  - Input shape: {synthetic_batch.shape}")
    
    # Extract embeddings
    embeddings = model.extract_embeddings(synthetic_batch, pooling='mean')
    
    print(f"  - Output shape: {embeddings.shape}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print(f"  - Mean: {embeddings.mean():.4f}")
    print(f"  - Std: {embeddings.std():.4f}")


def example_5_similarity_analysis():
    """Example 5: Compute similarity between embeddings."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Embedding Similarity Analysis")
    print("="*70)
    
    # Try to load embeddings for two tickers
    try:
        ticker1, ticker2 = 'AAPL', 'GOOGL'
        
        data1 = load_embeddings(ticker1, Config.EMBEDDINGS_SAVE_PATH)
        data2 = load_embeddings(ticker2, Config.EMBEDDINGS_SAVE_PATH)
        
        emb1 = data1['embeddings']
        emb2 = data2['embeddings']
        
        # Take first 100 embeddings from each
        n = min(100, len(emb1), len(emb2))
        emb1 = emb1[:n]
        emb2 = emb2[:n]
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Average embedding per ticker
        avg_emb1 = emb1.mean(axis=0, keepdims=True)
        avg_emb2 = emb2.mean(axis=0, keepdims=True)
        
        similarity = cosine_similarity(avg_emb1, avg_emb2)[0, 0]
        
        print(f"\nCosine similarity between {ticker1} and {ticker2}:")
        print(f"  - Similarity: {similarity:.4f}")
        print(f"  - Using {n} windows from each ticker")
        
        # Pairwise similarities
        pairwise_sim = cosine_similarity(emb1, emb2)
        
        print(f"\nPairwise similarities:")
        print(f"  - Shape: {pairwise_sim.shape}")
        print(f"  - Mean: {pairwise_sim.mean():.4f}")
        print(f"  - Std: {pairwise_sim.std():.4f}")
        print(f"  - Min: {pairwise_sim.min():.4f}")
        print(f"  - Max: {pairwise_sim.max():.4f}")
        
    except FileNotFoundError:
        print("\n✗ Embeddings not found. Please run training first.")
    except ImportError:
        print("\n✗ scikit-learn not installed. Install with: pip install scikit-learn")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PATCHTST USAGE EXAMPLES")
    print("="*70)
    
    # Example 1: Load model
    model = example_1_load_pretrained_model()
    
    # Example 2: Extract embeddings from new data
    example_2_extract_embeddings_from_data(model)
    
    # Example 3: Load saved embeddings
    example_3_load_saved_embeddings()
    
    # Example 4: Batch processing
    example_4_batch_processing(model)
    
    # Example 5: Similarity analysis
    example_5_similarity_analysis()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

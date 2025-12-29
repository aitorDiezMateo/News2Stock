"""
Utilities for extracting and saving LSTM embeddings
"""
import torch
import numpy as np
import pandas as pd
import os
from .config import Config

def extract_embeddings_from_loader(model, dataloader, device):
    """
    Extract embeddings from trained model using a dataloader
    
    Returns:
        embeddings: numpy array
        targets: numpy array
        dates: list of dates corresponding to each embedding
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            embeddings = model.get_embedding(X_batch)
            all_embeddings.append(embeddings.cpu().numpy())
    
    if not all_embeddings:
        return np.array([])
        
    return np.vstack(all_embeddings)

def save_embeddings(ticker, train_emb, val_emb, test_emb, train_dates, val_dates, test_dates, save_path):
    """
    Save all embeddings to a single Parquet file for the ticker.
    Compatible with news embeddings format.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Prepare dataframes for each split
    dfs = []
    
    if len(train_emb) > 0 and len(train_dates) == len(train_emb):
        df_train = pd.DataFrame({
            'Date': train_dates,
            'Ticker': ticker,
            'embedding': list(train_emb),
            'split': 'train'
        })
        dfs.append(df_train)
        
    if len(val_emb) > 0 and len(val_dates) == len(val_emb):
        df_val = pd.DataFrame({
            'Date': val_dates,
            'Ticker': ticker,
            'embedding': list(val_emb),
            'split': 'validation'
        })
        dfs.append(df_val)
        
    if len(test_emb) > 0 and len(test_dates) == len(test_emb):
        df_test = pd.DataFrame({
            'Date': test_dates,
            'Ticker': ticker,
            'embedding': list(test_emb),
            'split': 'test'
        })
        dfs.append(df_test)
    
    if not dfs:
        print(f"Warning: No embeddings to save for {ticker}")
        return
        
    # 2. Concatenate all splits
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 3. Sort by date
    full_df = full_df.sort_values('Date').reset_index(drop=True)
    
    # 4. Save to Parquet
    output_file = os.path.join(save_path, f"{ticker}_embeddings.parquet")
    full_df.to_parquet(output_file, index=False, engine='pyarrow')
    
    print(f"  ✓ Saved embeddings: {output_file}")
    print(f"    - Total rows: {len(full_df)}")
    print(f"    - Embedding dim: {len(full_df['embedding'].iloc[0])}")

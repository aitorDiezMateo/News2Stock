"""
Utilities for extracting and saving Chronos embeddings
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from chronos import ChronosPipeline

def load_chronos_model(model_name, device):
    """Load Chronos pipeline."""
    print(f"Loading {model_name} on {device}...")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print("Model loaded successfully.")
    return pipeline

def extract_embeddings(pipeline, dataset, batch_size=64):
    """
    Extract embeddings using Chronos pipeline.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    all_tickers = []
    all_dates = []
    idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Chronos embeddings"):
            # pipeline.embed expects tensor of shape (batch, seq_len)
            # batch is already a tensor from dataloader, usually on CPU
            # pipeline handles device movement internally usually, but let's be safe
            
            # The embed method returns (embeddings, scale) or just embeddings
            result = pipeline.embed(batch)
            
            if isinstance(result, tuple):
                batch_embs = result[0]
            else:
                batch_embs = result
            
            # Dimensions: (batch, seq_len, d_model)
            # We want one vector per window. Chronos embeddings are contextual.
            # Strategy: Mean pooling over the sequence dimension to get a robust representation
            if batch_embs.dim() == 3:
                batch_embs = batch_embs.mean(dim=1)
            
            # Convert to float32 (likely bfloat16 coming out of cuda)
            batch_embs = batch_embs.float().cpu().numpy()
            
            all_embeddings.append(batch_embs)
            
            # Collect metadata
            current_batch_size = len(batch)
            for i in range(current_batch_size):
                meta = dataset.get_metadata(idx)
                all_tickers.append(meta['ticker'])
                all_dates.append(meta['date'])
                idx += 1
                
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return {
        'embeddings': all_embeddings,
        'tickers': all_tickers,
        'dates': all_dates
    }

def save_embeddings(embeddings_dict, save_path):
    """
    Save embeddings to Parquet format, compatible with news embeddings.
    Structure: Date | Ticker | embedding (list)
    """
    os.makedirs(save_path, exist_ok=True)
    
    embeddings = embeddings_dict['embeddings']
    tickers = embeddings_dict['tickers']
    dates = embeddings_dict['dates']
    
    unique_tickers = sorted(list(set(tickers)))
    
    print(f"\nSaving embeddings to {save_path} (Format: Parquet)")
    
    for ticker in unique_tickers:
        # Filter for ticker
        mask = [t == ticker for t in tickers]
        ticker_embs = embeddings[mask]
        ticker_dates = np.array(dates)[mask]
        
        # Convert to list for dataframe
        emb_list = list(ticker_embs)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': ticker_dates,
            'Ticker': ticker,
            'embedding': emb_list
        })
        
        # Sort
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Save
        output_file = f"{save_path}{ticker}_embeddings.parquet"
        df.to_parquet(output_file, index=False, engine='pyarrow')
        
        print(f"  ✓ {ticker}: {len(df)} rows -> {output_file}")

"""
Utilities for extracting and saving embeddings
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os


def extract_embeddings(model, dataset, device, pooling='mean', batch_size=128):
    """
    Extract embeddings for all windows in the dataset.
    
    Args:
        model: Trained PatchTST model
        dataset: StockWindowDataset
        device: Device to run on
        pooling: Pooling strategy ('mean', 'max', 'last', 'cls')
        batch_size: Batch size for extraction
        
    Returns:
        Dictionary with embeddings, tickers, and dates
    """
    model.eval()
    
    all_embeddings = []
    all_tickers = []
    all_dates = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        idx = 0
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            batch = batch.to(device)
            
            # Extract embeddings
            embeddings = model.extract_embeddings(batch, pooling=pooling)
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Get metadata for this batch
            for i in range(len(batch)):
                metadata = dataset.get_metadata(idx)
                all_tickers.append(metadata['ticker'])
                all_dates.append(metadata['date'])
                idx += 1
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return {
        'embeddings': all_embeddings,
        'tickers': all_tickers,
        'dates': all_dates
    }


import pandas as pd

def save_embeddings_by_ticker(
    embeddings_dict, 
    save_path, 
    window_size, 
    target_col
):
    """
    Save embeddings grouped by ticker, compatible with news embeddings properly.
    
    Args:
        embeddings_dict: Dictionary from extract_embeddings()
        save_path: Directory to save embeddings
        window_size: Window size used
        target_col: Target column used
    """
    os.makedirs(save_path, exist_ok=True)
    
    embeddings = embeddings_dict['embeddings']
    tickers = embeddings_dict['tickers']
    dates = embeddings_dict['dates']
    
    unique_tickers = sorted(list(set(tickers)))
    
    print(f"\nSaving embeddings to {save_path} (Format: Parquet)")
    
    for ticker in unique_tickers:
        # Get indices for this ticker
        ticker_mask = np.array([t == ticker for t in tickers])
        ticker_embeddings = embeddings[ticker_mask]
        ticker_dates = np.array(dates)[ticker_mask]
        
        # Create DataFrame
        # Convert embeddings array to list of arrays for storage
        embeddings_list = list(ticker_embeddings)
        
        df = pd.DataFrame({
            'Date': ticker_dates,
            'Ticker': ticker,
            'embedding': embeddings_list
        })
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Save as parquet
        output_file = f"{save_path}{ticker}_embeddings.parquet"
        df.to_parquet(output_file, index=False, engine='pyarrow')
        
        print(f"  ✓ {ticker}: {len(df)} rows -> {output_file}")


def load_embeddings(ticker, embeddings_path):
    """
    Load embeddings for a specific ticker.
    
    Args:
        ticker: Stock ticker
        embeddings_path: Path to embeddings directory
        
    Returns:
        Dictionary with embeddings and metadata
    """
    file_path = f"{embeddings_path}{ticker}_embeddings.parquet"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings not found: {file_path}")
    
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    # Convert list column back to numpy array
    try:
        # Check if 'embedding' column exists
        if 'embedding' in df.columns:
            col_name = 'embedding'
        elif 'patchtst_embedding' in df.columns:
            col_name = 'patchtst_embedding'
        else:
            raise KeyError("No embedding column found")
            
        embeddings = np.stack(df[col_name].values)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        # Fallback for simple loading in case of issues
        embeddings = np.array(df[col_name].tolist())
        
    dates = df['Date'].values
    
    return {
        'embeddings': embeddings,
        'dates': dates,
        'ticker': ticker
    }

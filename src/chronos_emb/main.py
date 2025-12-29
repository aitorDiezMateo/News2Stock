"""
Main script for Chronos Embeddings Extraction
"""
import os
import sys

# Add src to path to allow absolute imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_emb.config import Config
from chronos_emb.dataset import StockWindowDataset
from chronos_emb.embeddings import load_chronos_model, extract_embeddings, save_embeddings

def main():
    Config.print_config()
    
    # 1. Create Dataset
    print("\n" + "="*70)
    print("STEP 1: Creating Dataset")
    print("="*70)
    
    dataset = StockWindowDataset(
        tickers=Config.TICKERS,
        window_size=Config.WINDOW_SIZE,
        target_col=Config.TARGET_COL,
        data_path=Config.DATA_PATH_LOAD,
        stride=Config.STRIDE
    )
    
    if len(dataset) == 0:
        print("Error: No data found.")
        return

    # 2. Load Model
    print("\n" + "="*70)
    print("STEP 2: Loading Model")
    print("="*70)
    
    pipeline = load_chronos_model(Config.MODEL_NAME, Config.DEVICE)
    
    # 3. Extract Embeddings
    print("\n" + "="*70)
    print("STEP 3: Extracting Embeddings")
    print("="*70)
    
    embeddings_dict = extract_embeddings(
        pipeline, 
        dataset, 
        batch_size=Config.BATCH_SIZE
    )
    
    print(f"\nExtracted shape: {embeddings_dict['embeddings'].shape}")
    
    # 4. Save Embeddings
    print("\n" + "="*70)
    print("STEP 4: Saving Results")
    print("="*70)
    
    save_embeddings(embeddings_dict, Config.EMBEDDINGS_SAVE_PATH)
    
    print("\n" + "="*70)
    print("PROCESS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

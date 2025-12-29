"""
Main script for LSTM Multi-Head Embeddings Pipeline
Loads data, trains model per ticker, extracts embeddings, and saves them.
"""
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_emb.config import Config
from lstm_emb.dataset import DataProcessor, create_sequences, StockDataset
from lstm_emb.model import StockLSTMMultiHead, WeightedMSELoss
from lstm_emb.trainer import train_model
from lstm_emb.embeddings import extract_embeddings_from_loader, save_embeddings

def process_ticker(ticker):
    print(f"\n{'#'*80}")
    print(f"# Processing {ticker}")
    print('#'*80)
    
    # 1. Load and Process Data
    processor = DataProcessor(ticker)
    data = processor.load_and_process()
    
    if data is None:
        print(f"Skipping {ticker}")
        return
        
    train_df = data['train']
    val_df = data['val']
    test_df = data['test']
    features = data['features']
    
    print(f"Features: {len(features)}")
    
    # 2. Create Sequences
    X_train, y_train, dates_train = create_sequences(train_df, features, Config.TARGETS, Config.SEQUENCE_LENGTH)
    X_val, y_val, dates_val = create_sequences(val_df, features, Config.TARGETS, Config.SEQUENCE_LENGTH)
    X_test, y_test, dates_test = create_sequences(test_df, features, Config.TARGETS, Config.SEQUENCE_LENGTH)
    
    if len(X_train) == 0:
        print(f"Not enough data for {ticker}")
        return

    # 3. Create DataLoaders
    # Note: Use drop_last=False for validation/test to ensure all embeddings are extracted
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    # Important: shuffle=False for val/test to keep order aligned with dates
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    input_size = len(features)
    output_size = len(Config.TARGETS)
    
    model = StockLSTMMultiHead(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        output_size=output_size,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    criterion = WeightedMSELoss(Config.TARGET_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.L2_REG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # 5. Train Model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        Config.EPOCHS, Config.PATIENCE, Config.DEVICE, scheduler
    )
    
    # 6. Extract Embeddings
    print("\nExtracting embeddings...")
    # Use loaders with shuffle=False (re-create train loader just to be safe for extraction order)
    train_loader_seq = DataLoader(StockDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    train_emb = extract_embeddings_from_loader(model, train_loader_seq, Config.DEVICE)
    val_emb = extract_embeddings_from_loader(model, val_loader, Config.DEVICE)
    test_emb = extract_embeddings_from_loader(model, test_loader, Config.DEVICE)
    
    # 7. Save Embeddings
    save_embeddings(
        ticker, 
        train_emb, val_emb, test_emb,
        dates_train, dates_val, dates_test,
        Config.EMBEDDINGS_SAVE_PATH
    )
    
    # 8. Save Model
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    model_path = os.path.join(Config.RESULTS_PATH, f"{ticker}_lstm_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

def main():
    Config.print_config()
    
    for ticker in Config.TICKERS:
        try:
            process_ticker(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

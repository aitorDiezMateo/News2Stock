"""
Training Script WITHOUT News Embeddings
========================================
Same as main.py but uses ONLY stock embeddings (no news).
This allows comparison to measure the impact of news information.

Usage:
    python main_no_news.py
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.config import Config
from prediction.dataset import (
    set_seed,
    load_stock_embeddings,
    load_stock_data,
    compute_target_label
)
from prediction.model import StockPredictionModel
from prediction.trainer import Trainer, evaluate_model
from prediction.metrics import (
    compute_metrics,
    print_metrics,
    save_metrics_to_csv
)


# =============================================================================
# DATASET WITHOUT NEWS
# =============================================================================

class StockOnlyDataset(Dataset):
    """Dataset using ONLY stock embeddings (no news)."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_dataset_no_news(
    tickers: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    """
    Create dataset with ONLY stock embeddings (no news).
    
    Returns:
        Tuple of (features, labels, ticker_indices, ticker_names, dates)
    """
    if tickers is None:
        tickers = Config.TICKERS
    
    all_features = []
    all_labels = []
    all_ticker_indices = []
    all_dates = []
    
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    for ticker in tickers:
        if verbose:
            print(f"\nProcessing {ticker}...")
        
        try:
            # Load stock data only
            stock_emb_df = load_stock_embeddings(ticker)
            stock_data_df = load_stock_data(ticker)
            
            if verbose:
                print(f"  Stock embeddings: {len(stock_emb_df)} windows")
            
            # Create date index for stock data
            stock_data_df = stock_data_df.set_index('Date')
            
            valid_samples = 0
            
            for idx, row in stock_emb_df.iterrows():
                window_end_date = row['Date']
                stock_embedding = np.array(row['embedding'])
                
                # Get next trading day data
                next_day_mask = stock_data_df.index > window_end_date
                if not next_day_mask.any():
                    continue
                
                next_day_idx = stock_data_df.index[next_day_mask][0]
                
                # Get current and next day prices
                if window_end_date not in stock_data_df.index:
                    continue
                    
                current_price = stock_data_df.loc[window_end_date, 'Close']
                next_price = stock_data_df.loc[next_day_idx, 'Close']
                volatility = stock_data_df.loc[window_end_date, 'VOLATILITY']
                
                # Compute target label
                label = compute_target_label(
                    current_price, 
                    next_price, 
                    volatility,
                    Config.NEUTRAL_THRESHOLD
                )
                
                # Features = stock embedding only (no news)
                if Config.INCLUDE_TICKER_FEATURE:
                    ticker_onehot = np.zeros(len(tickers))
                    ticker_onehot[ticker_to_idx[ticker]] = 1.0
                    features = np.concatenate([stock_embedding, ticker_onehot])
                else:
                    features = stock_embedding
                
                all_features.append(features)
                all_labels.append(label)
                all_ticker_indices.append(ticker_to_idx[ticker])
                all_dates.append(window_end_date)
                valid_samples += 1
            
            if verbose:
                print(f"  ✓ {valid_samples} samples")
                
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            continue
    
    if len(all_features) == 0:
        return np.array([]), np.array([]), np.array([]), tickers, []
    
    features = np.array(all_features)
    labels = np.array(all_labels)
    ticker_indices = np.array(all_ticker_indices)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Total samples: {len(features)}")
        print(f"Feature dimension: {features.shape[1]} (stock only, NO news)")
        print(f"Class distribution: DOWN={np.sum(labels==0)}, NEUTRAL={np.sum(labels==1)}, UP={np.sum(labels==2)}")
    
    return features, labels, ticker_indices, tickers, all_dates


def create_data_loaders_no_news(
    features: np.ndarray,
    labels: np.ndarray,
    ticker_indices: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
    temporal_split: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Create train/val/test data loaders with temporal split.
    """
    set_seed(seed)
    
    n = len(features)
    
    if temporal_split:
        # Temporal split (assumes data is sorted by date)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_features = features[:train_end]
        train_labels = labels[:train_end]
        
        val_features = features[train_end:val_end]
        val_labels = labels[train_end:val_end]
        
        test_features = features[val_end:]
        test_labels = labels[val_end:]
    else:
        # Random split
        indices = np.random.permutation(n)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        train_features, train_labels = features[train_idx], labels[train_idx]
        val_features, val_labels = features[val_idx], labels[val_idx]
        test_features, test_labels = features[test_idx], labels[test_idx]
    
    # Compute class weights from training set
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Create datasets
    train_dataset = StockOnlyDataset(train_features, train_labels)
    val_dataset = StockOnlyDataset(val_features, val_labels)
    test_dataset = StockOnlyDataset(test_features, test_labels)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, class_weights


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline WITHOUT news embeddings."""
    
    print("\n" + "=" * 70)
    print("STOCK PRICE MOVEMENT PREDICTION - NO NEWS EMBEDDINGS")
    print("Using ONLY stock embeddings (for comparison)")
    print("=" * 70)
    
    set_seed(Config.SEED)
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Stock embedding type: {Config.STOCK_EMBEDDING_TYPE}")
    print(f"  Stock embedding dim: {Config.STOCK_EMBEDDING_DIM}")
    print(f"  NEWS EMBEDDINGS: DISABLED")
    print(f"  Tickers: {Config.TICKERS}")
    
    # Calculate input dimension (stock only)
    input_dim = Config.STOCK_EMBEDDING_DIM
    if Config.INCLUDE_TICKER_FEATURE:
        input_dim += len(Config.TICKERS)
    
    print(f"  Input dimension: {input_dim}")
    
    # Create directories
    Config.create_directories()
    
    # ========================================================================
    # STEP 1: LOAD DATA (NO NEWS)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data (Stock Only)")
    print("=" * 70)
    
    features, labels, ticker_indices, tickers, dates = create_dataset_no_news(
        tickers=Config.TICKERS,
        verbose=True
    )
    
    if len(features) == 0:
        print("❌ No data available.")
        return
    
    # Sort by date for temporal split
    sorted_idx = np.argsort([d.value for d in dates])
    features = features[sorted_idx]
    labels = labels[sorted_idx]
    ticker_indices = ticker_indices[sorted_idx]
    dates = [dates[i] for i in sorted_idx]
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders_no_news(
        features=features,
        labels=labels,
        ticker_indices=ticker_indices,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        batch_size=Config.BATCH_SIZE,
        seed=Config.SEED,
        temporal_split=True
    )
    
    # ========================================================================
    # STEP 2: CREATE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Creating Model (smaller due to no news)")
    print("=" * 70)
    
    # Smaller hidden dims since input is smaller
    hidden_dims = [256, 128, 64]
    
    model = StockPredictionModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
        use_batch_norm=Config.USE_BATCH_NORM,
        activation=Config.ACTIVATION
    )
    
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # ========================================================================
    # STEP 3: TRAIN MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Training Model")
    print("=" * 70)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=Config.DEVICE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        use_scheduler=Config.USE_SCHEDULER,
        scheduler_patience=Config.SCHEDULER_PATIENCE,
        scheduler_factor=Config.SCHEDULER_FACTOR,
        min_lr=Config.MIN_LR,
        gradient_clip=Config.GRADIENT_CLIP,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_no_news.pt')
    
    history = trainer.train(
        num_epochs=Config.NUM_EPOCHS,
        save_path=model_path,
        verbose=True
    )
    
    # ========================================================================
    # STEP 4: EVALUATE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Evaluating Model")
    print("=" * 70)
    
    print("\nTest set results:")
    y_true, y_pred, y_proba = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=Config.DEVICE
    )
    
    test_metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(test_metrics, "Test Set Metrics (NO NEWS)")
    
    # Validation metrics
    y_val_true, y_val_pred, y_val_proba = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=Config.DEVICE
    )
    val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba)
    
    # ========================================================================
    # STEP 5: SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Saving Results")
    print("=" * 70)
    
    # Save metrics
    save_metrics_to_csv(
        test_metrics,
        os.path.join(Config.RESULTS_PATH, 'test_metrics_no_news.csv')
    )
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'NO_NEWS_EMBEDDINGS',
        'tickers': Config.TICKERS,
        'num_samples': len(features),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'num_epochs_trained': len(history['train_loss']),
        'best_val_loss': float(trainer.best_val_loss),
        'best_val_acc': float(trainer.best_val_acc),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro']),
        'news_embeddings': False
    }
    
    with open(os.path.join(Config.RESULTS_PATH, 'training_info_no_news.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE (NO NEWS)")
    print("=" * 70)
    
    print(f"\nResults WITHOUT news embeddings:")
    print(f"  Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro):    {test_metrics['f1_macro']:.4f}")
    print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for class_name in Config.CLASS_NAMES:
        f1 = test_metrics[f'{class_name.lower()}_f1']
        print(f"  {class_name}: {f1:.4f}")
    
    print(f"\nCompare with main.py (WITH news) to measure news impact!")
    print(f"\nSaved files:")
    print(f"  Model: {model_path}")
    print(f"  Metrics: {Config.RESULTS_PATH}test_metrics_no_news.csv")
    
    print("\n" + "=" * 70 + "\n")
    
    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main()

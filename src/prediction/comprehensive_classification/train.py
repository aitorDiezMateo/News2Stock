"""
Comprehensive Multimodal Experiment - Training
===============================================
Tests all combinations of features and tasks.
Generates results similar to finbert_lstm format.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from .config import Config
from .dataset import prepare_all_tickers, create_dataloaders
from .models import LSTMClassifier


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds) * 100,
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100,
    }
    
    return total_loss / len(data_loader), metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str,
    epochs: int = Config.EPOCHS
) -> Tuple[nn.Module, Dict]:
    """Train with early stopping."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    
    pbar = tqdm(range(epochs), desc=f'{model_name}', leave=False)
    
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix({'acc': f"{val_metrics['accuracy']:.1f}%"})
        
        if patience_counter >= Config.PATIENCE:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, {}


def get_input_dim(use_price, use_news, use_ts, use_sentiment):
    """Calculate input dimension."""
    dim = 0
    if use_price:
        dim += 7  # price features
    if use_news:
        dim += Config.NEWS_COMPRESSED_DIM
    if use_ts:
        dim += Config.TS_COMPRESSED_DIM
    if use_sentiment:
        dim += Config.SENTIMENT_DIM
    return dim


def run_experiment_for_ticker(
    ticker: str,
    ticker_samples: List[Dict],
    compute_sentiment: bool = False
) -> pd.DataFrame:
    """
    Run all experiments for a single ticker.
    
    Returns DataFrame with results for all configurations.
    """
    device = torch.device(Config.DEVICE)
    
    print(f"\n{'='*60}")
    print(f"TICKER: {ticker}")
    print(f"{'='*60}")
    print(f"Samples: {len(ticker_samples)}")
    
    # Define all configurations to test
    configurations = [
        # Format: (name, use_price, use_news, use_ts, use_sentiment)
        ('News+TS', False, True, True, False),
        ('News+TS+Price', True, True, True, False),
        ('News+Price', True, True, False, False),
        ('Price_Only', True, False, False, False),
    ]
    
    # Add sentiment-based if computed
    if compute_sentiment:
        configurations.extend([
            ('Sentiment+TS+Price', True, False, True, True),
            ('Sentiment+Price', True, False, False, True),
        ])
    
    tasks = ['3class', '2class']
    
    results = []
    
    for task in tasks:
        num_classes = 3 if task == '3class' else 2
        task_name = "3-Class (Up/Neutral/Down)" if task == '3class' else "2-Class (Neutral/Change)"
        
        print(f"\n{task_name}")
        print("-" * 60)
        
        for config_name, use_price, use_news, use_ts, use_sentiment in configurations:
            # Skip if sentiment not computed
            if use_sentiment and not compute_sentiment:
                continue
            
            # Get input dimension
            input_dim = get_input_dim(use_price, use_news, use_ts, use_sentiment)
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                ticker_samples,
                task=task,
                use_price=use_price,
                use_news=use_news,
                use_ts=use_ts,
                use_sentiment=use_sentiment,
                batch_size=Config.BATCH_SIZE
            )
            
            # Create model
            model = LSTMClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
            
            # Train
            model, _ = train_model(model, train_loader, val_loader, device, config_name)
            
            # Test
            criterion = nn.CrossEntropyLoss()
            test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
            
            result = {
                'ticker': ticker,
                'task': task,
                'configuration': config_name,
                'accuracy': test_metrics['accuracy'],
                'f1_score': test_metrics['f1'],
            }
            results.append(result)
            
            print(f"  {config_name:25s} Acc: {test_metrics['accuracy']:.2f}%  F1: {test_metrics['f1']:.2f}%")
    
    return pd.DataFrame(results)


def main():
    """Main entry point."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE CLASSIFICATION EXPERIMENT")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Tickers: {', '.join(Config.TICKERS)}")
    print("="*70)
    
    # Ask if sentiment should be computed
    print("\n⚠️  Computing FinBERT sentiment is SLOW (~10 min per ticker)")
    compute_sentiment = input("Compute sentiment? (y/N): ").strip().lower() == 'y'
    
    # Prepare data
    print("\nPreparing data...")
    all_ticker_data = prepare_all_tickers(compute_sentiment=compute_sentiment, verbose=True)
    
    # Run experiments for each ticker
    all_results = []
    
    for ticker in Config.TICKERS:
        ticker_results = run_experiment_for_ticker(
            ticker,
            all_ticker_data[ticker],
            compute_sentiment
        )
        all_results.append(ticker_results)
    
    # Combine results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    results_path = os.path.join(Config.RESULTS_PATH, 'comprehensive_results.csv')
    final_results.to_csv(results_path, index=False)
    
    print("\n" + "="*70)
    print("SUMMARY - ACCURACY BY CONFIGURATION")
    print("="*70)
    
    # Pivot tables
    for task in ['3class', '2class']:
        task_results = final_results[final_results['task'] == task]
        
        task_name = "3-Class (Up/Neutral/Down)" if task == '3class' else "2-Class (Neutral/Change)"
        print(f"\n{task_name}:")
        print("-" * 70)
        
        pivot = task_results.pivot(index='ticker', columns='configuration', values='accuracy')
        print(pivot.to_string())
        
        # Averages
        print("\nAverages:")
        avgs = task_results.groupby('configuration')['accuracy'].mean().sort_values(ascending=False)
        for config, avg_acc in avgs.items():
            print(f"  {config:25s}: {avg_acc:.2f}%")
    
    # Create FinBERT-like format
    print("\n" + "="*70)
    print("Creating FinBERT-style results table...")
    print("="*70)
    
    finbert_style_rows = []
    
    for ticker in Config.TICKERS:
        ticker_data = final_results[final_results['ticker'] == ticker]
        
        row = {'ticker': ticker}
        
        for _, result_row in ticker_data.iterrows():
            task = result_row['task']
            config = result_row['configuration']
            acc = result_row['accuracy']
            f1 = result_row['f1_score']
            
            # Create column names
            col_prefix = f"{config}_{task}"
            row[f'{col_prefix}_Accuracy'] = acc
            row[f'{col_prefix}_F1'] = f1
        
        finbert_style_rows.append(row)
    
    finbert_style_df = pd.DataFrame(finbert_style_rows)
    finbert_style_path = os.path.join(Config.RESULTS_PATH, 'comprehensive_results_finbert_style.csv')
    finbert_style_df.to_csv(finbert_style_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {finbert_style_path}")
    
    # Best configuration
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    
    for task in ['3class', '2class']:
        task_results = final_results[final_results['task'] == task]
        best_config = task_results.groupby('configuration')['accuracy'].mean().idxmax()
        best_acc = task_results.groupby('configuration')['accuracy'].mean().max()
        
        task_name = "3-Class" if task == '3class' else "2-Class"
        print(f"\n{task_name}: {best_config} ({best_acc:.2f}% avg accuracy)")
    
    return final_results


if __name__ == '__main__':
    main()

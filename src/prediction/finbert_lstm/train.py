"""
FinBERT-LSTM Experiment - Training Script
==========================================
Implements training and evaluation for all model architectures.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import Config
from .models import create_model, count_parameters
from .dataset import (
    get_sentiment_analyzer,
    prepare_dataset_with_sentiment,
    prepare_dataset_price_only,
    StockPriceDataset,
    create_data_loaders
)


class Trainer:
    """Trainer for stock price prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            batch_x = batch[0].to(self.device)
            batch_y = batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self, loader: DataLoader = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Validate model and return denormalized predictions."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        all_preds_actual = []
        all_targets_actual = []
        all_current_actual = []
        
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                batch_y = batch[1].to(self.device)
                batch_scaling = batch[2]  # [target_price, current_price, price_min, price_max]
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Denormalize predictions and targets
                preds_norm = outputs.cpu().numpy()
                scaling = batch_scaling.numpy()
                
                # Inverse transform: pred = pred_norm * (max - min) + min
                price_min = scaling[:, 2]
                price_max = scaling[:, 3]
                preds_actual = preds_norm * (price_max - price_min) + price_min
                targets_actual = scaling[:, 0]  # Already stored actual target price
                current_actual = scaling[:, 1]  # Actual current price
                
                all_preds_actual.append(preds_actual)
                all_targets_actual.append(targets_actual)
                all_current_actual.append(current_actual)
        
        preds_actual = np.concatenate(all_preds_actual)
        targets_actual = np.concatenate(all_targets_actual)
        current_actual = np.concatenate(all_current_actual)
        
        return total_loss / len(loader), preds_actual, targets_actual, current_actual
    
    def train(self, epochs: int = 100, patience: int = 20, verbose: bool = True) -> Dict:
        """Full training loop with early stopping."""
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, _, _, _ = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def evaluate(self) -> Dict:
        """Evaluate model on test set and compute metrics."""
        _, preds_actual, targets_actual, current_actual = self.validate(self.test_loader)
        
        # Predictions and targets are now in actual price space
        metrics = compute_metrics(preds_actual, targets_actual, current_actual)
        
        return metrics


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, current_prices: np.ndarray = None) -> Dict:
    """
    Compute evaluation metrics following the paper.
    
    Metrics:
        - MAE: Mean Absolute Error
        - MAPE: Mean Absolute Percentage Error
        - Accuracy: 1 - MAPE (as defined in paper)
        - RMSE: Root Mean Squared Error
        - Direction Accuracy: % of correct direction predictions
    """
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # MAPE (avoid division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100
    
    # Accuracy (as defined in paper: 1 - MAPE/100)
    accuracy = max(0, 1 - mape / 100) * 100
    
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Direction accuracy (if current prices provided)
    if current_prices is not None and len(targets) > 0:
        # Predicted direction: pred > current
        pred_up = predictions > current_prices
        true_up = targets > current_prices
        direction_acc = np.mean(pred_up == true_up) * 100
    elif len(targets) > 1:
        pred_direction = np.sign(np.diff(predictions))
        true_direction = np.sign(np.diff(targets))
        direction_acc = np.mean(pred_direction == true_direction) * 100
    else:
        direction_acc = 0
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'Accuracy': accuracy,
        'RMSE': rmse,
        'Direction_Accuracy': direction_acc
    }


def run_experiment(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: str = 'cuda',
    epochs: int = 100,
    verbose: bool = True
) -> Dict:
    """Run experiment for a single model."""
    
    # Create model
    model = create_model(model_type, input_dim=input_dim)
    num_params = count_parameters(model)
    
    if verbose:
        print(f"\n{model_type.upper()}: {num_params:,} parameters")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, test_loader, device)
    train_history = trainer.train(epochs=epochs, patience=Config.EARLY_STOPPING_PATIENCE, verbose=verbose)
    
    # Evaluate
    metrics = trainer.evaluate()
    
    if verbose:
        print(f"  MAE: ${metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%, "
              f"Acc: {metrics['Accuracy']:.2f}%, Dir_Acc: {metrics['Direction_Accuracy']:.2f}%")
    
    return {
        'model_type': model_type,
        'num_params': num_params,
        'metrics': metrics,
        'train_history': train_history
    }


def main():
    """Main function to run all experiments."""
    print("=" * 70)
    print("FinBERT-LSTM Experiment")
    print("Paper: Predicting Stock Prices with FinBERT-LSTM")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Sequence length: {Config.SEQUENCE_LENGTH} trading days")
    print(f"Tickers: {Config.TICKERS}")
    
    # Load FinBERT once for all tickers
    print("\nLoading FinBERT model...")
    tokenizer, model = get_sentiment_analyzer()
    use_sentiment = tokenizer is not None and model is not None
    
    if not use_sentiment:
        print("WARNING: FinBERT not available, skipping sentiment experiments")
    
    # Results storage
    all_results = []
    ticker_results = {}
    
    for ticker in Config.TICKERS:
        print(f"\n{'='*70}")
        print(f"TICKER: {ticker}")
        print(f"{'='*70}")
        
        ticker_results[ticker] = {}
        ticker_row = {'ticker': ticker}
        
        # ===== EXPERIMENT 1: FinBERT-LSTM (Sentiment + Price) =====
        if use_sentiment:
            print("\n" + "-"*50)
            print("Preparing FinBERT-LSTM data (with sentiment)...")
            
            try:
                sequences, targets, current_prices, dates = prepare_dataset_with_sentiment(
                    ticker, tokenizer, model
                )
                
                train_loader, val_loader, test_loader = create_data_loaders(
                    sequences, targets, current_prices,
                    for_lstm=True,
                    batch_size=Config.BATCH_SIZE
                )
                
                result = run_experiment(
                    model_type='finbert_lstm',
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    input_dim=4,  # 3 sentiment + 1 price
                    device=device,
                    epochs=Config.NUM_EPOCHS
                )
                ticker_results[ticker]['finbert_lstm'] = result['metrics']
                for k, v in result['metrics'].items():
                    ticker_row[f'finbert_lstm_{k}'] = v
                
            except Exception as e:
                print(f"  Error with FinBERT-LSTM: {e}")
                ticker_results[ticker]['finbert_lstm'] = None
        
        # ===== EXPERIMENT 2: Standard LSTM (Price only) =====
        print("\n" + "-"*50)
        print("Preparing Standard LSTM data (price only)...")
        
        try:
            sequences, targets, current_prices, dates = prepare_dataset_price_only(ticker)
            
            train_loader, val_loader, test_loader = create_data_loaders(
                sequences, targets, current_prices,
                for_lstm=True,
                batch_size=Config.BATCH_SIZE
            )
            
            result = run_experiment(
                model_type='lstm',
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                input_dim=1,
                device=device,
                epochs=Config.NUM_EPOCHS
            )
            ticker_results[ticker]['lstm'] = result['metrics']
            for k, v in result['metrics'].items():
                ticker_row[f'lstm_{k}'] = v
            
            # Store loaders for DNN (same data)
            loaders_price = (train_loader, val_loader, test_loader)
                
        except Exception as e:
            print(f"  Error with LSTM: {e}")
            ticker_results[ticker]['lstm'] = None
            loaders_price = None
        
        # ===== EXPERIMENT 3: DNN (Price only) =====
        print("\n" + "-"*50)
        print("Running DNN experiment (price only)...")
        
        if loaders_price is not None:
            try:
                train_loader, val_loader, test_loader = loaders_price
                
                result = run_experiment(
                    model_type='dnn',
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    input_dim=Config.SEQUENCE_LENGTH,
                    device=device,
                    epochs=Config.NUM_EPOCHS
                )
                ticker_results[ticker]['dnn'] = result['metrics']
                for k, v in result['metrics'].items():
                    ticker_row[f'dnn_{k}'] = v
                    
            except Exception as e:
                print(f"  Error with DNN: {e}")
                ticker_results[ticker]['dnn'] = None
        else:
            print("  Skipping DNN (no price data)")
        
        all_results.append(ticker_row)
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Aggregate results
    print("\n--- Average Metrics Across All Tickers ---\n")
    
    metrics = ['MAE', 'MAPE', 'Accuracy', 'Direction_Accuracy']
    models = ['finbert_lstm', 'lstm', 'dnn']
    
    summary_data = []
    for model_name in models:
        model_metrics = {'Model': model_name.upper()}
        for metric in metrics:
            col = f'{model_name}_{metric}'
            if col in results_df.columns:
                model_metrics[metric] = results_df[col].mean()
        summary_data.append(model_metrics)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    
    # Detailed results
    results_path = os.path.join(Config.RESULTS_PATH, 'finbert_lstm_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Summary
    summary_path = os.path.join(Config.RESULTS_PATH, 'finbert_lstm_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    # Print comparison
    print("\n--- Model Comparison (Paper vs Ours) ---")
    print("\nPaper reports:")
    print("  FinBERT-LSTM: MAPE ~3-5%, Accuracy ~95-97%")
    print("  LSTM: MAPE ~5-10%, Accuracy ~90-95%")
    print("  DNN: MAPE ~8-15%, Accuracy ~85-92%")
    print("\nNote: Paper uses different dataset (S&P 500 stocks)")
    
    return results_df, summary_df


if __name__ == "__main__":
    main()

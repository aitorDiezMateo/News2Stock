"""
Main Training Script for LSTM-based Stock Price Movement Prediction
=====================================================================
Strategy: Daily news aggregation + LSTM/GRU for temporal processing.

Usage:
    python main.py
"""
import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_lstm.config import Config
from prediction_lstm.dataset import (
    set_seed,
    create_combined_dataset,
    create_data_loaders
)
from prediction_lstm.model import LSTMPredictionModel
from prediction_lstm.trainer import Trainer, evaluate_model
from prediction_lstm.metrics import (
    compute_metrics,
    print_metrics,
    save_all_visualizations,
    save_metrics_to_csv
)


def main():
    """Main training pipeline."""
    
    print("\n" + "=" * 70)
    print("LSTM-BASED STOCK PRICE MOVEMENT PREDICTION")
    print("Strategy: Daily News Aggregation + LSTM/GRU")
    print("=" * 70)
    
    # Set seed
    set_seed(Config.SEED)
    
    # Print configuration
    Config.print_config()
    
    # Create directories
    Config.create_directories()
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data (Daily News Sequences)")
    print("=" * 70)
    
    stock_emb, news_seq, tech_feat, labels, ticker_idx, tickers, dates = create_combined_dataset(
        tickers=Config.TICKERS,
        verbose=True
    )
    
    if len(labels) == 0:
        print("❌ No data available.")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        stock_embeddings=stock_emb,
        news_sequences=news_seq,
        technical_features=tech_feat,
        labels=labels,
        ticker_indices=ticker_idx,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        batch_size=Config.BATCH_SIZE,
        seed=Config.SEED
    )
    
    print(f"\n  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # ========================================================================
    # STEP 2: CREATE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Creating LSTM Model")
    print("=" * 70)
    
    # Calculate total additional features dimension (price + technical)
    additional_features_dim = Config.ADDITIONAL_FEATURES_DIM
    
    model = LSTMPredictionModel(
        stock_embedding_dim=Config.STOCK_EMBEDDING_DIM,
        news_embedding_dim=Config.NEWS_EMBEDDING_DIM,
        technical_features_dim=additional_features_dim,
        lstm_hidden_size=Config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=Config.LSTM_NUM_LAYERS,
        lstm_dropout=Config.LSTM_DROPOUT,
        lstm_bidirectional=Config.LSTM_BIDIRECTIONAL,
        rnn_type=Config.RNN_TYPE,
        fusion_hidden_dims=Config.FUSION_HIDDEN_DIMS,
        num_classes=Config.NUM_CLASSES,
        fusion_dropout=Config.FUSION_DROPOUT,
        use_batch_norm=Config.USE_BATCH_NORM,
        include_ticker=Config.INCLUDE_TICKER_FEATURE,
        num_tickers=len(Config.TICKERS)
    )
    
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # ========================================================================
    # STEP 3: TRAIN
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
    
    model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pt')
    
    history = trainer.train(
        num_epochs=Config.NUM_EPOCHS,
        save_path=model_path,
        verbose=True
    )
    
    # ========================================================================
    # STEP 4: EVALUATE
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Evaluating Model")
    print("=" * 70)
    
    print("\nEvaluating on test set...")
    y_true, y_pred, y_proba = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=Config.DEVICE
    )
    
    test_metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(test_metrics, "Test Set Metrics")
    
    # ========================================================================
    # STEP 5: SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Saving Results")
    print("=" * 70)
    
    save_all_visualizations(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        history=history,
        save_dir=Config.PLOTS_PATH
    )
    
    save_metrics_to_csv(test_metrics, os.path.join(Config.RESULTS_PATH, 'test_metrics.csv'))
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'LSTM Daily Aggregation',
        'window_size': Config.WINDOW_SIZE,
        'rnn_type': Config.RNN_TYPE,
        'lstm_hidden_size': Config.LSTM_HIDDEN_SIZE,
        'lstm_bidirectional': Config.LSTM_BIDIRECTIONAL,
        'num_samples': len(labels),
        'train_samples': len(train_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'num_epochs_trained': len(history['train_loss']),
        'best_val_loss': float(trainer.best_val_loss),
        'best_val_acc': float(trainer.best_val_acc),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro']),
    }
    
    with open(os.path.join(Config.RESULTS_PATH, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for class_name in Config.CLASS_NAMES:
        f1 = test_metrics[f'{class_name.lower()}_f1']
        print(f"  {class_name}: {f1:.4f}")
    
    print(f"\nSaved Files:")
    print(f"  Model: {model_path}")
    print(f"  Plots: {Config.PLOTS_PATH}")
    print(f"  Results: {Config.RESULTS_PATH}")
    
    print("\n" + "=" * 70 + "\n")
    
    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main()

"""
Main training script for Attention-based Stock Prediction
=========================================================

Usage:
    python -m src.prediction_attention.main
    
    or from workspace root:
    python src/prediction_attention/main.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict

from src.prediction_attention.config import Config
from src.prediction_attention.dataset import (
    create_combined_dataset,
    create_data_loaders
)
from src.prediction_attention.model import AttentionPredictionModel
from src.prediction_attention.trainer import AttentionTrainer
from src.prediction_attention.metrics import (
    print_metrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_attention_weights,
    plot_attention_heatmap
)


def print_config():
    """Print configuration settings."""
    print("=" * 70)
    print("ATTENTION-BASED STOCK PREDICTION CONFIGURATION")
    print("=" * 70)
    print(f"\nData Configuration:")
    print(f"  - Tickers: {', '.join(Config.TICKERS)}")
    print(f"  - Window size: {Config.WINDOW_SIZE} days")
    print(f"  - Stock embedding type: {Config.STOCK_EMBEDDING_TYPE}")
    print(f"  - News embedding type: {Config.NEWS_EMBEDDING_TYPE}")
    
    print(f"\nEmbedding Dimensions:")
    print(f"  - Stock embedding: {Config.STOCK_EMBEDDING_DIM}")
    print(f"  - News embedding (per day): {Config.NEWS_EMBEDDING_DIM}")
    
    print(f"\nAttention Architecture:")
    print(f"  - Attention dim: {Config.ATTENTION_DIM}")
    print(f"  - Num attention heads: {Config.NUM_ATTENTION_HEADS}")
    print(f"  - Num transformer layers: {Config.NUM_TRANSFORMER_LAYERS}")
    print(f"  - Use positional encoding: {Config.USE_POSITIONAL_ENCODING}")
    print(f"  - Attention dropout: {Config.ATTENTION_DROPOUT}")
    
    print(f"\nFusion Network:")
    print(f"  - Hidden dims: {Config.FUSION_HIDDEN_DIMS}")
    print(f"  - Output classes: {Config.NUM_CLASSES}")
    
    print(f"\nTraining:")
    print(f"  - Batch size: {Config.BATCH_SIZE}")
    print(f"  - Epochs: {Config.NUM_EPOCHS}")
    print(f"  - Learning rate: {Config.LEARNING_RATE}")
    print(f"  - Device: {Config.DEVICE}")
    print("=" * 70)


def setup_directories():
    """Create necessary directories."""
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)


def create_model() -> AttentionPredictionModel:
    """Create model instance."""
    # Calculate total additional features dimension (price + technical)
    additional_features_dim = Config.ADDITIONAL_FEATURES_DIM
    
    model = AttentionPredictionModel(
        stock_embedding_dim=Config.STOCK_EMBEDDING_DIM,
        news_embedding_dim=Config.NEWS_EMBEDDING_DIM,
        technical_features_dim=additional_features_dim,
        attention_dim=Config.ATTENTION_DIM,
        num_attention_heads=Config.NUM_ATTENTION_HEADS,
        num_transformer_layers=Config.NUM_TRANSFORMER_LAYERS,
        attention_dropout=Config.ATTENTION_DROPOUT,
        use_positional_encoding=Config.USE_POSITIONAL_ENCODING,
        fusion_hidden_dims=Config.FUSION_HIDDEN_DIMS,
        num_classes=Config.NUM_CLASSES,
        fusion_dropout=Config.FUSION_DROPOUT,
        use_batch_norm=Config.USE_BATCH_NORM,
        include_ticker=Config.INCLUDE_TICKER_FEATURE,
        num_tickers=len(Config.TICKERS)
    )
    
    return model


def run_training():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("ATTENTION-BASED STOCK PRICE MOVEMENT PREDICTION")
    print("Strategy: Daily News Aggregation + Self-Attention")
    print("=" * 70)
    
    print_config()
    setup_directories()
    
    # Step 1: Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data (Daily News Sequences with Masks)")
    print("=" * 70)
    
    stock_emb, news_seq, news_masks, tech_feat, labels, ticker_idx, tickers, dates = create_combined_dataset(
        tickers=Config.TICKERS,
        verbose=True
    )
    
    if len(labels) == 0:
        print("❌ No data available.")
        return
    
    # Step 2: Create data loaders
    print("\n" + "=" * 70)
    print("STEP 2: Creating Data Loaders")
    print("=" * 70)
    
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        stock_embeddings=stock_emb,
        news_sequences=news_seq,
        news_masks=news_masks,
        technical_features=tech_feat,
        labels=labels,
        ticker_indices=ticker_idx,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        batch_size=Config.BATCH_SIZE,
        seed=Config.SEED
    )
    
    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 3: Create model
    print("\n" + "=" * 70)
    print("STEP 3: Creating Attention Model")
    print("=" * 70)
    
    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 3: Create model
    print("\n" + "=" * 70)
    print("STEP 3: Creating Attention Model")
    print("=" * 70)
    
    model = create_model()
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Step 4: Training
    print("\n" + "=" * 70)
    print("STEP 4: Training Model")
    print("=" * 70)
    
    trainer = AttentionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        scheduler_type=Config.SCHEDULER_TYPE,
        patience=Config.EARLY_STOPPING_PATIENCE,
        class_weights=class_weights if Config.USE_CLASS_WEIGHTS else None,
        save_dir=Config.MODEL_SAVE_DIR
    )
    
    training_results = trainer.train(
        num_epochs=Config.NUM_EPOCHS,
        save_checkpoints=True,
        checkpoint_freq=Config.CHECKPOINT_FREQUENCY
    )
    
    # Step 5: Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 5: Evaluating Model")
    print("=" * 70)
    
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print_metrics(test_metrics, "Test Set Results")
    
    # Get predictions for visualization
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    device = trainer.device
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch - order depends on config
            if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                stock_emb_b, news_seq_b, news_mask_b, tech_feat_b, ticker_b, labels_b = batch
                tech_feat_b = tech_feat_b.to(device)
                ticker_b = ticker_b.to(device)
            elif Config.USE_TECHNICAL_FEATURES:
                stock_emb_b, news_seq_b, news_mask_b, tech_feat_b, labels_b = batch
                tech_feat_b = tech_feat_b.to(device)
                ticker_b = None
            elif Config.INCLUDE_TICKER_FEATURE:
                stock_emb_b, news_seq_b, news_mask_b, ticker_b, labels_b = batch
                tech_feat_b = None
                ticker_b = ticker_b.to(device)
            else:
                stock_emb_b, news_seq_b, news_mask_b, labels_b = batch
                tech_feat_b = None
                ticker_b = None
            
            stock_emb_b = stock_emb_b.to(device)
            news_seq_b = news_seq_b.to(device)
            news_mask_b = news_mask_b.to(device)
            
            logits = model(stock_emb_b, news_seq_b, news_mask_b, tech_feat_b, ticker_b)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_b.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Step 6: Save results and plots
    print("\n" + "=" * 70)
    print("STEP 6: Saving Results")
    print("=" * 70)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels,
        all_preds,
        save_path=os.path.join(Config.PLOTS_DIR, 'confusion_matrix.png')
    )
    
    # Plot training history
    plot_training_history(
        training_results['history'],
        save_path=os.path.join(Config.PLOTS_DIR, 'training_history.png')
    )
    
    # Attention analysis
    print("\nAnalyzing attention patterns...")
    attention_analysis = trainer.get_attention_analysis(test_loader, num_samples=200)
    
    # Plot attention patterns
    plot_attention_weights(
        attention_analysis['attention_weights'],
        attention_analysis['masks'],
        attention_analysis['labels'],
        attention_analysis['predictions'],
        save_path=os.path.join(Config.PLOTS_DIR, 'attention_patterns.png')
    )
    
    plot_attention_heatmap(
        attention_analysis['attention_weights'],
        attention_analysis['labels'],
        save_path=os.path.join(Config.PLOTS_DIR, 'attention_heatmap.png')
    )
    
    # Save results JSON
    results = {
        'config': {
            'stock_embedding_type': Config.STOCK_EMBEDDING_TYPE,
            'news_embedding_type': Config.NEWS_EMBEDDING_TYPE,
            'window_size': Config.WINDOW_SIZE,
            'attention_dim': Config.ATTENTION_DIM,
            'num_attention_heads': Config.NUM_ATTENTION_HEADS,
            'num_transformer_layers': Config.NUM_TRANSFORMER_LAYERS,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'include_ticker': Config.INCLUDE_TICKER_FEATURE,
            'use_positional_encoding': Config.USE_POSITIONAL_ENCODING,
        },
        'training': {
            'epochs_trained': training_results['epochs_trained'],
            'training_time': training_results['training_time'],
            'best_val_loss': training_results['best_val_loss'],
            'best_val_accuracy': training_results['best_val_accuracy'],
        },
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision_macro': test_metrics['precision_macro'],
            'recall_macro': test_metrics['recall_macro'],
            'f1_macro': test_metrics['f1_macro'],
            'confusion_matrix': test_metrics['confusion_matrix'],
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(Config.RESULTS_DIR, 'attention_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {results_path}")
    
    # Save test metrics CSV
    import pandas as pd
    metrics_df = pd.DataFrame([{
        'model': 'AttentionPrediction',
        'accuracy': test_metrics['accuracy'],
        'precision_macro': test_metrics['precision_macro'],
        'recall_macro': test_metrics['recall_macro'],
        'f1_macro': test_metrics['f1_macro'],
        'roc_auc_macro': test_metrics.get('roc_auc_macro'),
        'epochs_trained': training_results['epochs_trained'],
        'training_time': training_results['training_time'],
    }])
    
    csv_path = os.path.join(Config.RESULTS_DIR, 'test_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"\nPer-Class F1 Scores:")
    for i, name in enumerate(Config.CLASS_NAMES):
        key = f'{name}_accuracy'
        if key in test_metrics:
            print(f"  {name}: {test_metrics[key]:.4f}")
    
    print(f"\nSaved Files:")
    print(f"  Model: {Config.MODEL_SAVE_DIR}")
    print(f"  Plots: {Config.PLOTS_DIR}")
    print(f"  Results: {Config.RESULTS_DIR}")
    print("=" * 70)
    
    return model, test_metrics


if __name__ == "__main__":
    model, metrics = run_training()

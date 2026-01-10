"""
Main Training Script for Stock Price Movement Prediction
==========================================================
Trains a neural network to predict whether stock prices go up, down, or stay neutral.

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

from prediction.config import Config
from prediction.dataset import (
    set_seed,
    create_combined_dataset,
    create_data_loaders
)
from prediction.model import StockPredictionModel
from prediction.trainer import Trainer, evaluate_model
from prediction.metrics import (
    compute_metrics,
    print_metrics,
    save_all_visualizations,
    save_metrics_to_csv
)


def main():
    """Main training pipeline."""
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print("\n" + "=" * 70)
    print("STOCK PRICE MOVEMENT PREDICTION")
    print("3-Class Classification: DOWN, NEUTRAL, UP")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Print configuration
    Config.print_config()
    
    # Create output directories
    Config.create_directories()
    
    # ========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 70)
    
    # Create combined dataset
    features, labels, ticker_indices, tickers, dates = create_combined_dataset(
        tickers=Config.TICKERS,
        verbose=True
    )
    
    if len(features) == 0:
        print("❌ No data available. Please check embedding files.")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        features=features,
        labels=labels,
        ticker_indices=ticker_indices,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        batch_size=Config.BATCH_SIZE,
        seed=Config.SEED
    )
    
    # Get labels for each split (for visualization)
    train_labels = train_loader.dataset.labels.numpy()
    val_labels = val_loader.dataset.labels.numpy()
    test_labels = test_loader.dataset.labels.numpy()
    
    # ========================================================================
    # STEP 2: CREATE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Creating Model")
    print("=" * 70)
    
    model = StockPredictionModel(
        input_dim=Config.get_input_dim(),
        hidden_dims=Config.HIDDEN_DIMS,
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
    
    # Create trainer
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
    
    # Model save path
    model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pt')
    
    # Train
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
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_true, y_pred, y_proba = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=Config.DEVICE
    )
    
    # Compute metrics
    test_metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(test_metrics, "Test Set Metrics")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_true, y_val_pred, y_val_proba = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=Config.DEVICE
    )
    val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba)
    print_metrics(val_metrics, "Validation Set Metrics")
    
    # ========================================================================
    # STEP 5: SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Saving Results")
    print("=" * 70)
    
    # Save visualizations
    save_all_visualizations(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        history=history,
        save_dir=Config.PLOTS_PATH,
        y_train=train_labels,
        y_val=val_labels
    )
    
    # Save metrics to CSV
    save_metrics_to_csv(
        test_metrics,
        os.path.join(Config.RESULTS_PATH, 'test_metrics.csv')
    )
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'tickers': Config.TICKERS,
        'training_mode': Config.TRAINING_MODE,
        'num_samples': len(features),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'input_dim': Config.get_input_dim(),
        'hidden_dims': Config.HIDDEN_DIMS,
        'num_epochs_trained': len(history['train_loss']),
        'best_val_loss': float(trainer.best_val_loss),
        'best_val_acc': float(trainer.best_val_acc),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro']),
        'class_weights': class_weights.tolist()
    }
    
    with open(os.path.join(Config.RESULTS_PATH, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✓ Training info saved to {Config.RESULTS_PATH}training_info.json")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro):  {test_metrics['f1_macro']:.4f}")
    print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    
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

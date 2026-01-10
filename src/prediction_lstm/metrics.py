"""
Metrics and Visualization for Stock Prediction
===============================================
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Optional
import os

from .config import Config


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """Compute classification metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    for i, class_name in enumerate(Config.CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        metrics[f'{class_name.lower()}_precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f'{class_name.lower()}_recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f'{class_name.lower()}_f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # ROC AUC if probabilities available
    if y_proba is not None:
        try:
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            metrics['roc_auc_macro'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """Print metrics summary."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):{metrics['f1_weighted']:.4f}")
    print(f"\n  Per-class F1:")
    for class_name in Config.CLASS_NAMES:
        f1 = metrics.get(f'{class_name.lower()}_f1', 0)
        print(f"    {class_name}: {f1:.4f}")


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(range(len(Config.CLASS_NAMES)))
    ax.set_yticks(range(len(Config.CLASS_NAMES)))
    ax.set_xticklabels(Config.CLASS_NAMES)
    ax.set_yticklabels(Config.CLASS_NAMES)
    
    # Add values
    for i in range(len(Config.CLASS_NAMES)):
        for j in range(len(Config.CLASS_NAMES)):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.colorbar(im)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def save_all_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    history: Dict,
    save_dir: str
):
    """Save all visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    plot_confusion_matrix(y_true, y_pred, os.path.join(save_dir, 'confusion_matrix.png'))


def save_metrics_to_csv(metrics: Dict, filepath: str):
    """Save metrics to CSV."""
    import pandas as pd
    
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")

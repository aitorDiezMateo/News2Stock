"""
Metrics utilities for Attention-based Stock Prediction Model
============================================================
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os

from .config import Config


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    class_names = Config.CLASS_NAMES
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_pred = y_pred[class_mask]
            metrics[f'{class_name}_accuracy'] = (class_pred == i).mean()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    
    # ROC AUC if probabilities available
    if y_prob is not None:
        try:
            metrics['roc_auc_macro'] = roc_auc_score(
                y_true, y_prob,
                multi_class='ovr',
                average='macro'
            )
            metrics['roc_auc_weighted'] = roc_auc_score(
                y_true, y_prob,
                multi_class='ovr',
                average='weighted'
            )
        except Exception:
            metrics['roc_auc_macro'] = None
            metrics['roc_auc_weighted'] = None
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    
    if metrics.get('roc_auc_macro') is not None:
        print(f"  ROC AUC (macro):    {metrics['roc_auc_macro']:.4f}")
    
    print(f"\n{metrics['classification_report']}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix - Attention Model"
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    class_names = Config.CLASS_NAMES
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    plt.close()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = "Training History - Attention Model"
):
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved: {save_path}")
    
    plt.close()


def plot_attention_weights(
    attention_weights: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    num_samples: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize attention patterns for different prediction outcomes.
    
    Args:
        attention_weights: (num_samples, window_size)
        masks: (num_samples, window_size)
        labels: True labels
        predictions: Model predictions
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    class_names = Config.CLASS_NAMES
    window_size = attention_weights.shape[1]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for class_idx, ax in enumerate(axes):
        # Find samples for this class
        class_mask = labels == class_idx
        class_attn = attention_weights[class_mask][:num_samples]
        class_masks = masks[class_mask][:num_samples]
        
        if len(class_attn) == 0:
            ax.set_title(f'{class_names[class_idx]} (No samples)')
            continue
        
        # Average attention
        avg_attn = class_attn.mean(axis=0)
        
        # Plot
        days = list(range(-window_size + 1, 1))
        ax.bar(days, avg_attn, alpha=0.7, color=['blue', 'green', 'red'][class_idx])
        ax.set_xlabel('Days before prediction')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'{class_names[class_idx]} - Average Attention Pattern')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention patterns saved: {save_path}")
    
    plt.close()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    labels: np.ndarray,
    num_samples: int = 50,
    save_path: Optional[str] = None
):
    """
    Create heatmap visualization of attention weights.
    """
    class_names = Config.CLASS_NAMES
    window_size = attention_weights.shape[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for class_idx, ax in enumerate(axes):
        class_mask = labels == class_idx
        class_attn = attention_weights[class_mask][:num_samples]
        
        if len(class_attn) == 0:
            ax.set_title(f'{class_names[class_idx]} (No samples)')
            continue
        
        sns.heatmap(
            class_attn,
            ax=ax,
            cmap='YlOrRd',
            xticklabels=5,
            yticklabels=False
        )
        ax.set_xlabel('Day Index')
        ax.set_ylabel('Sample')
        ax.set_title(f'{class_names[class_idx]} Attention Heatmap')
    
    plt.suptitle('Attention Weights by Class')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved: {save_path}")
    
    plt.close()

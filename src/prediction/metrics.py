"""
Metrics and Visualizations for Stock Price Movement Prediction
===============================================================
Comprehensive evaluation metrics and visualization tools for 3-class classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional
import os
import pandas as pd

from .config import Config


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    for i, class_name in enumerate(Config.CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        metrics[f'{class_name.lower()}_precision'] = precision_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f'{class_name.lower()}_recall'] = recall_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f'{class_name.lower()}_f1'] = f1_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Metrics"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*46}")
    
    for class_name in Config.CLASS_NAMES:
        cn = class_name.lower()
        print(f"  {class_name:<10} "
              f"{metrics[f'{cn}_precision']:<12.4f} "
              f"{metrics[f'{cn}_recall']:<12.4f} "
              f"{metrics[f'{cn}_f1']:<12.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  {'':>10}", end='')
    for name in Config.CLASS_NAMES:
        print(f"{name:>10}", end='')
    print()
    
    for i, row in enumerate(cm):
        print(f"  {Config.CLASS_NAMES[i]:>10}", end='')
        for val in row:
            print(f"{val:>10}", end='')
        print()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=Config.CLASS_NAMES,
        yticklabels=Config.CLASS_NAMES,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Classification Report"
) -> plt.Figure:
    """
    Plot classification report as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    report = classification_report(
        y_true, y_pred,
        target_names=Config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Select only class rows
    df = df.loc[Config.CLASS_NAMES + ['macro avg', 'weighted avg']]
    df = df[['precision', 'recall', 'f1-score']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.heatmap(
        df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Score'}
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy History')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution for all splits.
    
    Args:
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    splits = [('Train', y_train), ('Validation', y_val), ('Test', y_test)]
    
    for ax, (name, y) in zip(axes, splits):
        counts = [np.sum(y == i) for i in range(Config.NUM_CLASSES)]
        percentages = [c / len(y) * 100 for c in counts]
        
        bars = ax.bar(Config.CLASS_NAMES, counts, color=Config.CLASS_COLORS, edgecolor='black')
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.annotate(f'{pct:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'{name} Set (n={len(y)})')
        ax.set_ylabel('Count')
        ax.set_xlabel('Class')
    
    plt.suptitle('Class Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for each class (One-vs-Rest).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities [n_samples, n_classes]
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = Config.CLASS_COLORS
    
    for i, (class_name, color) in enumerate(zip(Config.CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curves for each class.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = Config.CLASS_COLORS
    
    for i, (class_name, color) in enumerate(zip(Config.CLASS_NAMES, colors)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f'{class_name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot prediction distribution compared to true distribution.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(Config.CLASS_NAMES))
    width = 0.35
    
    true_counts = [np.sum(y_true == i) for i in range(Config.NUM_CLASSES)]
    pred_counts = [np.sum(y_pred == i) for i in range(Config.NUM_CLASSES)]
    
    bars1 = ax.bar(x - width/2, true_counts, width, label='True', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', color='coral', edgecolor='black')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('True vs Predicted Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(Config.CLASS_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def save_all_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    history: Dict,
    save_dir: str,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
):
    """
    Save all visualizations to directory.
    
    Args:
        y_true: True test labels
        y_pred: Predicted test labels
        y_proba: Predicted probabilities
        history: Training history
        save_dir: Directory to save plots
        y_train: Training labels (for distribution plot)
        y_val: Validation labels (for distribution plot)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Saving Visualizations")
    print('='*60)
    
    # Training history
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    # Confusion matrix (normalized)
    plot_confusion_matrix(y_true, y_pred, normalize=True,
                         save_path=os.path.join(save_dir, 'confusion_matrix_normalized.png'))
    plt.close()
    
    # Confusion matrix (raw counts)
    plot_confusion_matrix(y_true, y_pred, normalize=False,
                         save_path=os.path.join(save_dir, 'confusion_matrix_counts.png'))
    plt.close()
    
    # Classification report
    plot_classification_report(y_true, y_pred,
                              save_path=os.path.join(save_dir, 'classification_report.png'))
    plt.close()
    
    # ROC curves
    plot_roc_curves(y_true, y_proba,
                   save_path=os.path.join(save_dir, 'roc_curves.png'))
    plt.close()
    
    # Precision-Recall curves
    plot_precision_recall_curves(y_true, y_proba,
                                save_path=os.path.join(save_dir, 'precision_recall_curves.png'))
    plt.close()
    
    # Prediction distribution
    plot_prediction_distribution(y_true, y_pred,
                                save_path=os.path.join(save_dir, 'prediction_distribution.png'))
    plt.close()
    
    # Class distribution (if train/val provided)
    if y_train is not None and y_val is not None:
        plot_class_distribution(y_train, y_val, y_true,
                               save_path=os.path.join(save_dir, 'class_distribution.png'))
        plt.close()
    
    print(f"\n✓ All visualizations saved to {save_dir}")


def save_metrics_to_csv(
    metrics: Dict,
    save_path: str
):
    """Save metrics to CSV file."""
    # Flatten metrics (exclude confusion matrix)
    flat_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    
    df = pd.DataFrame([flat_metrics])
    df.to_csv(save_path, index=False)
    print(f"✓ Metrics saved to {save_path}")


if __name__ == "__main__":
    # Test metrics visualization
    np.random.seed(42)
    
    # Dummy data
    y_true = np.random.randint(0, 3, 200)
    y_pred = np.random.randint(0, 3, 200)
    y_proba = np.random.rand(200, 3)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Compute and print metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics, "Test Metrics")
    
    # Test plots
    print("\nTesting plots...")
    fig = plot_confusion_matrix(y_true, y_pred)
    plt.show()

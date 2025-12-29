"""
Visualization utilities for PatchTST training
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    # Find best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('PatchTST MLM Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add text with best performance
    plt.text(
        0.02, 0.98, 
        f'Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history: {save_path}")
    
    plt.close()


def plot_learning_rate(lr_history, save_path=None):
    """
    Plot learning rate schedule.
    
    Args:
        lr_history: List of learning rates per epoch
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(lr_history) + 1)
    plt.plot(epochs, lr_history, 'b-', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved LR schedule: {save_path}")
    
    plt.close()


def plot_loss_distribution(train_losses, val_losses, save_path=None):
    """
    Plot distribution of losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss distribution
    ax1.hist(train_losses, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(train_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(train_losses):.4f}')
    ax1.set_xlabel('Loss', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Training Loss Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Validation loss distribution
    ax2.hist(val_losses, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(np.mean(val_losses), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_losses):.4f}')
    ax2.set_xlabel('Loss', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Validation Loss Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved loss distribution: {save_path}")
    
    plt.close()


def plot_reconstruction_examples(model, dataset, device, num_examples=4, save_path=None):
    """
    Plot examples of original vs reconstructed patches.
    
    Args:
        model: Trained PatchTST model
        dataset: Dataset to sample from
        device: Device to run on
        num_examples: Number of examples to plot
        save_path: Optional path to save the plot
    """
    import torch
    
    model.eval()
    
    # Sample random windows
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            window = dataset[idx].unsqueeze(0).to(device)
            
            # Get reconstruction
            output = model(window, return_embeddings=False, training=False)
            original = output['original_patches'][0].cpu().numpy()
            reconstructed = output['reconstruction'][0].cpu().numpy()
            
            # Flatten patches for visualization
            original_flat = original.flatten()
            reconstructed_flat = reconstructed.flatten()
            
            x = np.arange(len(original_flat))
            ax.plot(x, original_flat, 'b-', label='Original', linewidth=2, alpha=0.7)
            ax.plot(x, reconstructed_flat, 'r--', label='Reconstructed', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'Example {idx}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reconstruction examples: {save_path}")
    
    plt.close()

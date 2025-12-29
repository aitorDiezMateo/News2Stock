"""
Training utilities for PatchTST
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        
    Returns:
        LR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, device, gradient_clip=1.0):
    """
    Train for one epoch.
    
    Args:
        model: PatchTST model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        gradient_clip: Max gradient norm for clipping
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        output = model(batch, return_embeddings=False, training=True)
        loss = model.compute_loss(output)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, device):
    """
    Validate the model.
    
    Args:
        model: PatchTST model
        dataloader: Validation dataloader
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            batch = batch.to(device)
            
            output = model(batch, return_embeddings=False, training=True)
            loss = model.compute_loss(output)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath, config=None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        filepath: Path to save checkpoint
        config: Optional config dict
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

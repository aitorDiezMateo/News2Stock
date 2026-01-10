"""
Training utilities for Stock Price Movement Prediction
=======================================================
Includes trainer class with early stopping, learning rate scheduling,
and comprehensive logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import time

from .config import Config


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Trainer class for stock prediction model.
    
    Features:
        - Training and validation loops
        - Early stopping
        - Learning rate scheduling
        - Gradient clipping
        - Class-weighted loss
        - Checkpoint saving
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_scheduler: bool = True,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 20
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            class_weights: Optional class weights for imbalanced data
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            use_scheduler: Whether to use LR scheduler
            scheduler_patience: Patience for LR reduction
            scheduler_factor: Factor to reduce LR by
            min_lr: Minimum learning rate
            gradient_clip: Max gradient norm
            early_stopping_patience: Patience for early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        
        # Loss function with optional class weights
        if class_weights is not None and Config.USE_CLASS_WEIGHTS:
            class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=min_lr
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Maximum number of epochs
            save_path: Path to save best model
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING START")
            print(f"{'='*70}")
            print(f"Device: {self.device}")
            print(f"Max epochs: {num_epochs}")
            print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Learning rate scheduling
            if self.use_scheduler:
                self.scheduler.step(val_loss)
            
            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_loss, val_acc)
            
            # Early stopping check
            if self.early_stopping(val_loss):
                if verbose:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Logging
            if verbose:
                epoch_time = time.time() - epoch_start
                is_best = "✓ BEST" if epoch == self.best_epoch else ""
                
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s {is_best}")
        
        # Training complete
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Best epoch: {self.best_epoch + 1}")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            print(f"Best val accuracy: {self.best_val_acc:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print("✓ Loaded best model weights")
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        val_acc: float
    ):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'num_classes': self.model.num_classes,
                'dropout': self.model.dropout_rate,
                'use_batch_norm': self.model.use_batch_norm,
                'activation': self.model.activation_name
            }
        }
        
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_checkpoint(path: str, model: nn.Module, device: str = 'cpu') -> Dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            model: Model to load weights into
            device: Device to load on
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and get predictions.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device
        
    Returns:
        Tuple of (true_labels, predictions, probabilities)
    """
    model.eval()
    model.to(device)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    for X, y in dataloader:
        X = X.to(device)
        
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_labels.extend(y.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


if __name__ == "__main__":
    # Quick test of trainer components
    from .model import StockPredictionModel
    
    print("Testing Trainer components...")
    
    # Create dummy data
    X = torch.randn(100, Config.get_input_dim())
    y = torch.randint(0, 3, (100,))
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = StockPredictionModel()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device='cpu',
        learning_rate=1e-3
    )
    
    # Test one epoch
    train_loss, train_acc = trainer.train_epoch()
    val_loss, val_acc = trainer.validate()
    
    print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

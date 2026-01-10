"""
Trainer for LSTM-based Stock Price Movement Prediction
=======================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

from .config import Config


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
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
    """Trainer class for LSTM prediction model."""
    
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
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        
        # Loss function
        if class_weights is not None and Config.USE_CLASS_WEIGHTS:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=min_lr
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Tracking
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            # Unpack batch - order depends on config (tech features, ticker)
            if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                stock_emb, news_seq, tech_feat, ticker, labels = batch
                tech_feat = tech_feat.to(self.device)
                ticker = ticker.to(self.device)
            elif Config.USE_TECHNICAL_FEATURES:
                stock_emb, news_seq, tech_feat, labels = batch
                tech_feat = tech_feat.to(self.device)
                ticker = None
            elif Config.INCLUDE_TICKER_FEATURE:
                stock_emb, news_seq, ticker, labels = batch
                tech_feat = None
                ticker = ticker.to(self.device)
            else:
                stock_emb, news_seq, labels = batch
                tech_feat = None
                ticker = None
            
            stock_emb = stock_emb.to(self.device)
            news_seq = news_seq.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(stock_emb, news_seq, tech_feat, ticker)
            
            loss = self.criterion(logits, labels)
            loss.backward()
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch - order depends on config
                if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, tech_feat, ticker, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker = ticker.to(self.device)
                elif Config.USE_TECHNICAL_FEATURES:
                    stock_emb, news_seq, tech_feat, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker = None
                elif Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, ticker, labels = batch
                    tech_feat = None
                    ticker = ticker.to(self.device)
                else:
                    stock_emb, news_seq, labels = batch
                    tech_feat = None
                    ticker = None
                
                stock_emb = stock_emb.to(self.device)
                news_seq = news_seq.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(stock_emb, news_seq, tech_feat, ticker)
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """Full training loop."""
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_loss, val_acc)
            
            # Scheduler step
            if self.use_scheduler:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if self.early_stopping(val_loss):
                if verbose:
                    print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                break
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.2e}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        torch.save(checkpoint, path)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and return predictions."""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_proba = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch - order depends on config
            if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                stock_emb, news_seq, tech_feat, ticker, labels = batch
                tech_feat = tech_feat.to(device)
                ticker = ticker.to(device)
            elif Config.USE_TECHNICAL_FEATURES:
                stock_emb, news_seq, tech_feat, labels = batch
                tech_feat = tech_feat.to(device)
                ticker = None
            elif Config.INCLUDE_TICKER_FEATURE:
                stock_emb, news_seq, ticker, labels = batch
                tech_feat = None
                ticker = ticker.to(device)
            else:
                stock_emb, news_seq, labels = batch
                tech_feat = None
                ticker = None
            
            stock_emb = stock_emb.to(device)
            news_seq = news_seq.to(device)
            
            logits = model(stock_emb, news_seq, tech_feat, ticker)
            
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_proba)

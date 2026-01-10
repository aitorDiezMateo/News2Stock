"""
Trainer for Attention-based Stock Prediction Model
==================================================
Handles training loop, validation, and early stopping.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

from .config import Config
from .model import AttentionPredictionModel
from .metrics import compute_metrics


class AttentionTrainer:
    """
    Trainer for AttentionPredictionModel.
    """
    
    def __init__(
        self,
        model: AttentionPredictionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = 'auto',
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY,
        scheduler_type: str = Config.SCHEDULER_TYPE,
        patience: int = Config.EARLY_STOPPING_PATIENCE,
        min_delta: float = Config.EARLY_STOPPING_MIN_DELTA,
        class_weights: Optional[torch.Tensor] = None,
        save_dir: str = None
    ):
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=patience // 2
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2
            )
        else:
            self.scheduler = None
        
        self.scheduler_type = scheduler_type
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Save directory
        self.save_dir = save_dir or Config.MODEL_SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            # Unpack batch - order depends on config
            if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                stock_emb, news_seq, news_mask, tech_feat, ticker_onehot, labels = batch
                tech_feat = tech_feat.to(self.device)
                ticker_onehot = ticker_onehot.to(self.device)
            elif Config.USE_TECHNICAL_FEATURES:
                stock_emb, news_seq, news_mask, tech_feat, labels = batch
                tech_feat = tech_feat.to(self.device)
                ticker_onehot = None
            elif Config.INCLUDE_TICKER_FEATURE:
                stock_emb, news_seq, news_mask, ticker_onehot, labels = batch
                tech_feat = None
                ticker_onehot = ticker_onehot.to(self.device)
            else:
                stock_emb, news_seq, news_mask, labels = batch
                tech_feat = None
                ticker_onehot = None
            
            stock_emb = stock_emb.to(self.device)
            news_seq = news_seq.to(self.device)
            news_mask = news_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(stock_emb, news_seq, news_mask, tech_feat, ticker_onehot)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, loader: DataLoader = None) -> Tuple[float, float]:
        """Validate model."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                # Unpack batch - order depends on config
                if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, news_mask, tech_feat, ticker_onehot, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker_onehot = ticker_onehot.to(self.device)
                elif Config.USE_TECHNICAL_FEATURES:
                    stock_emb, news_seq, news_mask, tech_feat, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker_onehot = None
                elif Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, news_mask, ticker_onehot, labels = batch
                    tech_feat = None
                    ticker_onehot = ticker_onehot.to(self.device)
                else:
                    stock_emb, news_seq, news_mask, labels = batch
                    tech_feat = None
                    ticker_onehot = None
                
                stock_emb = stock_emb.to(self.device)
                news_seq = news_seq.to(self.device)
                news_mask = news_mask.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(stock_emb, news_seq, news_mask, tech_feat, ticker_onehot)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _check_early_stopping(self, val_loss: float, val_acc: float) -> bool:
        """Check if training should stop early."""
        improved = False
        
        # Check for improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_accuracy = val_acc
            self.epochs_without_improvement = 0
            self.best_model_state = self.model.state_dict().copy()
            improved = True
        else:
            self.epochs_without_improvement += 1
        
        return self.epochs_without_improvement >= self.patience
    
    def train(
        self,
        num_epochs: int = Config.NUM_EPOCHS,
        save_checkpoints: bool = True,
        checkpoint_freq: int = 10
    ) -> Dict:
        """
        Full training loop.
        """
        print(f"\n{'='*60}")
        print(f"Training AttentionPredictionModel")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self._train_epoch()
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Update scheduler
            if self.scheduler is not None:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            should_stop = self._check_early_stopping(val_loss, val_acc)
            
            if should_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
            
            # Save checkpoint
            if save_checkpoints and epoch % checkpoint_freq == 0:
                self._save_checkpoint(epoch)
        
        # Training complete
        training_time = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Time: {training_time}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_accuracy:.4f}")
        print(f"{'='*60}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save best model
        self._save_best_model()
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'training_time': str(training_time),
            'epochs_trained': len(self.history['train_loss'])
        }
    
    def evaluate(self, loader: DataLoader = None) -> Dict:
        """
        Full evaluation with detailed metrics.
        """
        if loader is None:
            loader = self.test_loader if self.test_loader is not None else self.val_loader
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                # Unpack batch - order depends on config
                if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, news_mask, tech_feat, ticker_onehot, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker_onehot = ticker_onehot.to(self.device)
                elif Config.USE_TECHNICAL_FEATURES:
                    stock_emb, news_seq, news_mask, tech_feat, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker_onehot = None
                elif Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, news_mask, ticker_onehot, labels = batch
                    tech_feat = None
                    ticker_onehot = ticker_onehot.to(self.device)
                else:
                    stock_emb, news_seq, news_mask, labels = batch
                    tech_feat = None
                    ticker_onehot = None
                
                stock_emb = stock_emb.to(self.device)
                news_seq = news_seq.to(self.device)
                news_mask = news_mask.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(stock_emb, news_seq, news_mask, tech_feat, ticker_onehot)
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_prob=np.array(all_probs)
        )
        
        return metrics
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        path = os.path.join(self.save_dir, f'attention_checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }, path)
        print(f"  Checkpoint saved: {path}")
    
    def _save_best_model(self):
        """Save best model."""
        path = os.path.join(self.save_dir, 'best_attention_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'stock_embedding_dim': self.model.stock_embedding_dim,
                'attention_dim': self.model.news_attention.attention_dim,
                'num_heads': self.model.news_attention.num_heads,
                'fusion_hidden_dims': self.model.fusion_hidden_dims,
                'num_classes': self.model.num_classes,
                'include_ticker': self.model.include_ticker,
            },
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
        }, path)
        print(f"Best model saved: {path}")
    
    def load_model(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {path}")
    
    def get_attention_analysis(self, loader: DataLoader, num_samples: int = 100) -> Dict:
        """
        Analyze attention patterns for interpretability.
        """
        self.model.eval()
        all_attention_weights = []
        all_masks = []
        all_labels = []
        all_preds = []
        
        count = 0
        with torch.no_grad():
            for batch in loader:
                if count >= num_samples:
                    break
                
                # Unpack batch - order depends on config
                if Config.USE_TECHNICAL_FEATURES and Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, news_mask, tech_feat, ticker_onehot, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker_onehot = ticker_onehot.to(self.device)
                elif Config.USE_TECHNICAL_FEATURES:
                    stock_emb, news_seq, news_mask, tech_feat, labels = batch
                    tech_feat = tech_feat.to(self.device)
                    ticker_onehot = None
                elif Config.INCLUDE_TICKER_FEATURE:
                    stock_emb, news_seq, news_mask, ticker_onehot, labels = batch
                    tech_feat = None
                    ticker_onehot = ticker_onehot.to(self.device)
                else:
                    stock_emb, news_seq, news_mask, labels = batch
                    tech_feat = None
                    ticker_onehot = None
                
                stock_emb = stock_emb.to(self.device)
                news_seq = news_seq.to(self.device)
                news_mask = news_mask.to(self.device)
                
                # Get attention weights
                attn_weights = self.model.get_attention_weights(news_seq, news_mask)
                
                # Get predictions
                logits = self.model(stock_emb, news_seq, news_mask, tech_feat, ticker_onehot)
                _, preds = torch.max(logits, 1)
                
                all_attention_weights.extend(attn_weights.cpu().numpy())
                all_masks.extend(news_mask.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())
                
                count += len(labels)
        
        return {
            'attention_weights': np.array(all_attention_weights[:num_samples]),
            'masks': np.array(all_masks[:num_samples]),
            'labels': np.array(all_labels[:num_samples]),
            'predictions': np.array(all_preds[:num_samples])
        }

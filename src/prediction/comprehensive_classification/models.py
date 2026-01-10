"""
Comprehensive Multimodal Experiment - Models
=============================================
Simple LSTM classifier with dynamic input dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class LSTMClassifier(nn.Module):
    """
    LSTM classifier with attention.
    Works with any combination of features.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_size: int = Config.LSTM_HIDDEN_SIZE,
        num_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        
        Returns:
            logits: [batch, num_classes]
        """
        # Project input
        x = self.input_proj(x)  # [B, S, 128]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [B, S, hidden*2]
        
        # Attention
        attn_weights = self.attention(lstm_out)  # [B, S, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # [B, hidden*2]
        
        # Classify
        logits = self.classifier(context)  # [B, num_classes]
        
        return logits

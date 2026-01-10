"""
FinBERT-LSTM Experiment - Models
=================================
Three architectures from the paper:
1. FinBERT-LSTM: Sentiment + Close prices
2. LSTM: Only close prices
3. DNN: Only close prices

Plus enhanced variant:
4. FinBERT-LSTM-EMB: Sentiment + Close prices + Stock embeddings
"""
import torch
import torch.nn as nn
from typing import List

from .config import Config


class FinBERTLSTM(nn.Module):
    """
    FinBERT-LSTM Architecture (from paper)
    
    Input: [batch, seq_len, 4]
        - 3 sentiment features (positive, negative, neutral)
        - 1 normalized close price
    
    Architecture:
        - 3 LSTM layers with 50 units each
        - Fully connected layer for output
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # 3 sentiment + 1 price
        hidden_size: int = 50,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch] predicted prices
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Output
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class FinBERTLSTMWithEmbeddings(nn.Module):
    """
    Enhanced FinBERT-LSTM with Stock Embeddings
    
    Input: [batch, seq_len, 4 + embedding_dim]
        - 3 sentiment features
        - 1 normalized close price
        - N embedding features (from Chronos)
    """
    
    def __init__(
        self,
        sentiment_dim: int = 3,
        price_dim: int = 1,
        embedding_dim: int = 768,
        hidden_size: int = 50,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = sentiment_dim + price_dim + embedding_dim
        
        # Project embeddings to smaller dimension
        self.embedding_proj = nn.Linear(embedding_dim, 32)
        
        # New input dim after projection
        lstm_input_dim = sentiment_dim + price_dim + 32
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
        self.sentiment_dim = sentiment_dim
        self.price_dim = price_dim
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, sentiment_dim + price_dim + embedding_dim]
        """
        # Split features
        sentiment = x[:, :, :self.sentiment_dim]
        price = x[:, :, self.sentiment_dim:self.sentiment_dim + self.price_dim]
        embedding = x[:, :, self.sentiment_dim + self.price_dim:]
        
        # Project embeddings
        embedding_proj = self.embedding_proj(embedding)
        
        # Concatenate
        combined = torch.cat([sentiment, price, embedding_proj], dim=-1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(combined)
        last_hidden = lstm_out[:, -1, :]
        
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class StandardLSTM(nn.Module):
    """
    Standard LSTM Architecture (from paper)
    
    Input: [batch, seq_len, 1] - only close prices
    
    Architecture:
        - 3 LSTM layers with 50 units each
        - Fully connected layer for output
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 50,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 1]
        Returns:
            [batch] predicted prices
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class DNN(nn.Module):
    """
    Deep Neural Network Architecture (from paper)
    
    Input: [batch, seq_len] - flattened close prices
    
    Architecture:
        - Batch normalization layer
        - 3 fully connected layers: 256 -> 128 -> 64
        - Output layer
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # seq_len flattened
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] or [batch, seq_len, 1]
        Returns:
            [batch] predicted prices
        """
        # Flatten if needed
        if len(x.shape) == 3:
            x = x.squeeze(-1)
        
        # Normalize
        x = self.batch_norm(x)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Output
        out = self.output_layer(x)
        return out.squeeze(-1)


def create_model(model_type: str, input_dim: int = None, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'finbert_lstm', 'finbert_lstm_emb', 'lstm', 'dnn'
        input_dim: Input dimension (varies by model type)
    """
    if model_type == 'finbert_lstm':
        return FinBERTLSTM(
            input_dim=input_dim or 4,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            **kwargs
        )
    elif model_type == 'finbert_lstm_emb':
        return FinBERTLSTMWithEmbeddings(
            sentiment_dim=3,
            price_dim=1,
            embedding_dim=Config.STOCK_EMBEDDING_DIM,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            **kwargs
        )
    elif model_type == 'lstm':
        return StandardLSTM(
            input_dim=1,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            **kwargs
        )
    elif model_type == 'dnn':
        return DNN(
            input_dim=Config.SEQUENCE_LENGTH,
            hidden_dims=Config.DNN_HIDDEN_DIMS,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing models...\n")
    
    batch_size = 16
    seq_len = 8
    
    # Test FinBERT-LSTM
    model = create_model('finbert_lstm', input_dim=4)
    x = torch.randn(batch_size, seq_len, 4)
    out = model(x)
    print(f"FinBERT-LSTM: input={x.shape} -> output={out.shape}, params={count_parameters(model):,}")
    
    # Test Standard LSTM
    model = create_model('lstm')
    x = torch.randn(batch_size, seq_len, 1)
    out = model(x)
    print(f"Standard LSTM: input={x.shape} -> output={out.shape}, params={count_parameters(model):,}")
    
    # Test DNN
    model = create_model('dnn')
    x = torch.randn(batch_size, seq_len)
    out = model(x)
    print(f"DNN: input={x.shape} -> output={out.shape}, params={count_parameters(model):,}")
    
    # Test FinBERT-LSTM with embeddings
    model = create_model('finbert_lstm_emb')
    x = torch.randn(batch_size, seq_len, 4 + 768)
    out = model(x)
    print(f"FinBERT-LSTM-EMB: input={x.shape} -> output={out.shape}, params={count_parameters(model):,}")

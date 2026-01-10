"""
LSTM/GRU Model for Stock Price Movement Prediction
===================================================
Processes daily news sequences through LSTM/GRU, then fuses with stock embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .config import Config


class NewsLSTM(nn.Module):
    """
    LSTM/GRU for processing daily news sequence.
    
    Input: (batch, window_size, news_embedding_dim)
    Output: (batch, lstm_output_dim)
    """
    
    def __init__(
        self,
        input_dim: int = Config.NEWS_EMBEDDING_DIM,
        hidden_size: int = Config.LSTM_HIDDEN_SIZE,
        num_layers: int = Config.LSTM_NUM_LAYERS,
        dropout: float = Config.LSTM_DROPOUT,
        bidirectional: bool = Config.LSTM_BIDIRECTIONAL,
        rnn_type: str = Config.RNN_TYPE
    ):
        super(NewsLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.rnn_type = rnn_type.lower()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # RNN layer
        RNN = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        self.output_dim = hidden_size * self.num_directions
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            (batch, output_dim) - final hidden state
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Run through RNN
        output, hidden = self.rnn(x)
        
        # Get final hidden state
        if self.rnn_type == 'lstm':
            hidden = hidden[0]  # (num_layers * num_directions, batch, hidden_size)
        
        # Concatenate forward and backward final hidden states
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch, hidden_size)
            # Get last layer's forward and backward
            forward_hidden = hidden[-2]  # (batch, hidden_size)
            backward_hidden = hidden[-1]  # (batch, hidden_size)
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = hidden[-1]  # (batch, hidden_size)
        
        # Project output
        output = self.output_proj(final_hidden)
        
        return output


class LSTMPredictionModel(nn.Module):
    """
    Full model: NewsLSTM + Stock Embedding Fusion + Classification.
    
    Architecture:
    1. Process news sequence through LSTM -> news_vector
    2. Concatenate: [stock_embedding, news_vector, ticker_onehot]
    3. Feed through fusion MLP for classification
    """
    
    def __init__(
        self,
        stock_embedding_dim: int = Config.STOCK_EMBEDDING_DIM,
        news_embedding_dim: int = Config.NEWS_EMBEDDING_DIM,
        technical_features_dim: int = 0,
        lstm_hidden_size: int = Config.LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = Config.LSTM_NUM_LAYERS,
        lstm_dropout: float = Config.LSTM_DROPOUT,
        lstm_bidirectional: bool = Config.LSTM_BIDIRECTIONAL,
        rnn_type: str = Config.RNN_TYPE,
        fusion_hidden_dims: List[int] = None,
        num_classes: int = Config.NUM_CLASSES,
        fusion_dropout: float = Config.FUSION_DROPOUT,
        use_batch_norm: bool = Config.USE_BATCH_NORM,
        include_ticker: bool = Config.INCLUDE_TICKER_FEATURE,
        num_tickers: int = len(Config.TICKERS)
    ):
        super(LSTMPredictionModel, self).__init__()
        
        if fusion_hidden_dims is None:
            fusion_hidden_dims = Config.FUSION_HIDDEN_DIMS
        
        self.stock_embedding_dim = stock_embedding_dim
        self.technical_features_dim = technical_features_dim
        self.include_ticker = include_ticker
        self.num_tickers = num_tickers
        self.num_classes = num_classes
        self.fusion_hidden_dims = fusion_hidden_dims
        self.fusion_dropout = fusion_dropout
        self.use_batch_norm = use_batch_norm
        
        # News LSTM
        self.news_lstm = NewsLSTM(
            input_dim=news_embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=lstm_bidirectional,
            rnn_type=rnn_type
        )
        
        lstm_output_dim = self.news_lstm.output_dim
        
        # Calculate fusion input dimension
        fusion_input_dim = stock_embedding_dim + lstm_output_dim
        if technical_features_dim > 0:
            fusion_input_dim += technical_features_dim
        if include_ticker:
            fusion_input_dim += num_tickers
        
        self.fusion_input_dim = fusion_input_dim
        
        # Build fusion network
        layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(fusion_dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fusion_network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.fusion_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        stock_embedding: torch.Tensor,
        news_sequence: torch.Tensor,
        technical_features: Optional[torch.Tensor] = None,
        ticker_onehot: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            stock_embedding: (batch, stock_embedding_dim)
            news_sequence: (batch, window_size, news_embedding_dim)
            technical_features: (batch, technical_features_dim) optional
            ticker_onehot: (batch, num_tickers) optional
            
        Returns:
            logits: (batch, num_classes)
        """
        # Process news sequence through LSTM
        news_vector = self.news_lstm(news_sequence)  # (batch, lstm_output_dim)
        
        # Concatenate features
        features = [stock_embedding, news_vector]
        if technical_features is not None and self.technical_features_dim > 0:
            features.append(technical_features)
        if self.include_ticker and ticker_onehot is not None:
            features.append(ticker_onehot)
        
        combined = torch.cat(features, dim=1)
        
        # Classification
        logits = self.fusion_network(combined)
        
        return logits
    
    def predict_proba(
        self,
        stock_embedding: torch.Tensor,
        news_sequence: torch.Tensor,
        ticker_onehot: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(stock_embedding, news_sequence, ticker_onehot)
        return F.softmax(logits, dim=1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        lines = [
            "LSTMPredictionModel(",
            f"  Stock embedding dim: {self.stock_embedding_dim}",
            f"  LSTM output dim: {self.news_lstm.output_dim}",
            f"  Fusion input dim: {self.fusion_input_dim}",
            f"  Fusion hidden dims: {self.fusion_hidden_dims}",
            f"  Output classes: {self.num_classes}",
            f"  Include ticker: {self.include_ticker}",
            f"  Total parameters: {self.count_parameters():,}",
            ")"
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("Testing LSTMPredictionModel...")
    
    batch_size = 32
    window_size = Config.WINDOW_SIZE
    
    model = LSTMPredictionModel()
    print(model)
    
    # Test inputs
    stock_emb = torch.randn(batch_size, Config.STOCK_EMBEDDING_DIM)
    news_seq = torch.randn(batch_size, window_size, Config.NEWS_EMBEDDING_DIM)
    ticker = torch.zeros(batch_size, len(Config.TICKERS))
    ticker[:, 0] = 1.0
    
    # Forward pass
    logits = model(stock_emb, news_seq, ticker)
    print(f"\nInput shapes:")
    print(f"  Stock: {stock_emb.shape}")
    print(f"  News sequence: {news_seq.shape}")
    print(f"  Ticker: {ticker.shape}")
    print(f"\nOutput shape: {logits.shape}")
    
    proba = model.predict_proba(stock_emb, news_seq, ticker)
    print(f"Probability sum (should be ~1.0): {proba[0].sum():.4f}")
    
    print("\n✓ Model test passed!")

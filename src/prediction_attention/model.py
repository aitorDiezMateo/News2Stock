"""
Attention Model for Stock Price Movement Prediction
====================================================
Processes daily news sequences through Self-Attention mechanism.
Key advantage: Learns to focus on important days, ignoring noise/filler news.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

from .config import Config


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalAttention(nn.Module):
    """
    Self-attention over daily news sequence.
    
    Learns which days are important for prediction.
    Can use mask to ignore days without news.
    """
    
    def __init__(
        self,
        input_dim: int = Config.NEWS_EMBEDDING_DIM,
        attention_dim: int = Config.ATTENTION_DIM,
        num_heads: int = Config.NUM_ATTENTION_HEADS,
        num_layers: int = Config.NUM_TRANSFORMER_LAYERS,
        dropout: float = Config.ATTENTION_DROPOUT,
        use_positional_encoding: bool = Config.USE_POSITIONAL_ENCODING,
        max_seq_len: int = 100
    ):
        super(TemporalAttention, self).__init__()
        
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.use_pos_enc = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(attention_dim, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention_dim,
            nhead=num_heads,
            dim_feedforward=attention_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False  # Disable to avoid padding issues
        )
        
        # Output dimension
        self.output_dim = attention_dim
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - daily news embeddings
            mask: (batch, seq_len) - 1 for valid days, 0 for empty days
            
        Returns:
            (batch, attention_dim) - attention-weighted summary
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, attention_dim)
        
        # Add positional encoding
        if self.use_pos_enc:
            x = self.pos_encoder(x)
        
        # Create attention mask for transformer (True = ignore)
        # Handle edge case: if all positions are masked, unmask all to prevent NaN
        if mask is not None:
            # Convert: 1 (valid) -> False (don't ignore), 0 (empty) -> True (ignore)
            src_key_padding_mask = (mask == 0)  # (batch, seq_len)
            
            # If any sample has ALL positions masked, unmask all for that sample
            all_masked = src_key_padding_mask.all(dim=1, keepdim=True)  # (batch, 1)
            src_key_padding_mask = src_key_padding_mask & ~all_masked  # Unmask all if all were masked
        else:
            src_key_padding_mask = None
        
        # Apply transformer - disable nested tensor optimization to avoid padding issues
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (batch, seq_len, attention_dim)
        
        # Simple mean pooling with mask instead of attention pooling (more stable)
        if mask is not None:
            # Expand mask for broadcasting
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            # Masked mean
            x_masked = x * mask_expanded
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
            output = x_masked.sum(dim=1) / sum_mask  # (batch, attention_dim)
        else:
            output = x.mean(dim=1)  # (batch, attention_dim)
        
        return output
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get approximate attention weights for visualization.
        Since we use masked mean pooling, we return the normalized mask.
        """
        if mask is not None:
            # Normalize mask to sum to 1
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
            weights = mask / mask_sum
        else:
            batch_size, seq_len = x.size(0), x.size(1)
            weights = torch.ones(batch_size, seq_len, device=x.device) / seq_len
        
        return weights  # (batch, seq_len)


class AttentionPredictionModel(nn.Module):
    """
    Full model: News Attention + Stock Embedding Fusion + Classification.
    
    Architecture:
    1. Process news sequence through Attention -> news_vector
    2. Concatenate: [stock_embedding, news_vector, ticker_onehot]
    3. Feed through fusion MLP for classification
    """
    
    def __init__(
        self,
        stock_embedding_dim: int = Config.STOCK_EMBEDDING_DIM,
        news_embedding_dim: int = Config.NEWS_EMBEDDING_DIM,
        technical_features_dim: int = 0,
        attention_dim: int = Config.ATTENTION_DIM,
        num_attention_heads: int = Config.NUM_ATTENTION_HEADS,
        num_transformer_layers: int = Config.NUM_TRANSFORMER_LAYERS,
        attention_dropout: float = Config.ATTENTION_DROPOUT,
        use_positional_encoding: bool = Config.USE_POSITIONAL_ENCODING,
        fusion_hidden_dims: List[int] = None,
        num_classes: int = Config.NUM_CLASSES,
        fusion_dropout: float = Config.FUSION_DROPOUT,
        use_batch_norm: bool = Config.USE_BATCH_NORM,
        include_ticker: bool = Config.INCLUDE_TICKER_FEATURE,
        num_tickers: int = len(Config.TICKERS)
    ):
        super(AttentionPredictionModel, self).__init__()
        
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
        
        # News Attention module
        self.news_attention = TemporalAttention(
            input_dim=news_embedding_dim,
            attention_dim=attention_dim,
            num_heads=num_attention_heads,
            num_layers=num_transformer_layers,
            dropout=attention_dropout,
            use_positional_encoding=use_positional_encoding
        )
        
        attention_output_dim = self.news_attention.output_dim
        
        # Calculate fusion input dimension
        fusion_input_dim = stock_embedding_dim + attention_output_dim
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
        news_mask: Optional[torch.Tensor] = None,
        technical_features: Optional[torch.Tensor] = None,
        ticker_onehot: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            stock_embedding: (batch, stock_embedding_dim)
            news_sequence: (batch, window_size, news_embedding_dim)
            news_mask: (batch, window_size) - 1 for days with news
            technical_features: (batch, technical_features_dim) optional
            ticker_onehot: (batch, num_tickers) optional
            
        Returns:
            logits: (batch, num_classes)
        """
        # Process news sequence through attention
        news_vector = self.news_attention(news_sequence, news_mask)  # (batch, attention_dim)
        
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
        news_mask: Optional[torch.Tensor] = None,
        ticker_onehot: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(stock_embedding, news_sequence, news_mask, ticker_onehot)
        return F.softmax(logits, dim=1)
    
    def get_attention_weights(
        self,
        news_sequence: torch.Tensor,
        news_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get attention weights for interpretability."""
        return self.news_attention.get_attention_weights(news_sequence, news_mask)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        lines = [
            "AttentionPredictionModel(",
            f"  Stock embedding dim: {self.stock_embedding_dim}",
            f"  Attention output dim: {self.news_attention.output_dim}",
            f"  Fusion input dim: {self.fusion_input_dim}",
            f"  Fusion hidden dims: {self.fusion_hidden_dims}",
            f"  Output classes: {self.num_classes}",
            f"  Include ticker: {self.include_ticker}",
            f"  Total parameters: {self.count_parameters():,}",
            ")"
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("Testing AttentionPredictionModel...")
    
    batch_size = 32
    window_size = Config.WINDOW_SIZE
    
    model = AttentionPredictionModel()
    print(model)
    
    # Test inputs
    stock_emb = torch.randn(batch_size, Config.STOCK_EMBEDDING_DIM)
    news_seq = torch.randn(batch_size, window_size, Config.NEWS_EMBEDDING_DIM)
    news_mask = torch.ones(batch_size, window_size)
    news_mask[:, -5:] = 0  # Last 5 days have no news
    ticker = torch.zeros(batch_size, len(Config.TICKERS))
    ticker[:, 0] = 1.0
    
    # Forward pass
    logits = model(stock_emb, news_seq, news_mask, ticker)
    print(f"\nInput shapes:")
    print(f"  Stock: {stock_emb.shape}")
    print(f"  News sequence: {news_seq.shape}")
    print(f"  News mask: {news_mask.shape}")
    print(f"  Ticker: {ticker.shape}")
    print(f"\nOutput shape: {logits.shape}")
    
    proba = model.predict_proba(stock_emb, news_seq, news_mask, ticker)
    print(f"Probability sum (should be ~1.0): {proba[0].sum():.4f}")
    
    # Test attention weights
    attn_weights = model.get_attention_weights(news_seq, news_mask)
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"Sample attention (should sum to ~1.0): {attn_weights[0].sum():.4f}")
    
    print("\n✓ Model test passed!")

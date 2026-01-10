"""
Stock Prediction Model
======================
Neural network for 3-class stock price movement prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class StockPredictionModel(nn.Module):
    """
    Feed-forward neural network for stock price movement prediction.
    
    Architecture:
        - Multiple hidden layers with configurable dimensions
        - Batch normalization (optional)
        - Dropout for regularization
        - Configurable activation function
        - 3-class output (DOWN, NEUTRAL, UP)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = 'leaky_relu'
    ):
        """
        Initialize the model.
        
        Args:
            input_dim: Dimension of input features (stock + news embeddings + technical features)
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128, 64])
            num_classes: Number of output classes (default: 3 for DOWN/NEUTRAL/UP)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'gelu', 'selu')
        """
        super(StockPredictionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        
        # Get activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        
        return activations[activation.lower()]
    
    def _initialize_weights(self):
        """Initialize network weights using appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # Initialize batch norm
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted classes.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        lines = [
            "StockPredictionModel(",
            f"  Input dim: {self.input_dim}",
            f"  Hidden dims: {self.hidden_dims}",
            f"  Output classes: {self.num_classes}",
            f"  Activation: {self.activation_name}",
            f"  Batch norm: {self.use_batch_norm}",
            f"  Dropout: {self.dropout}",
            f"  Total parameters: {self.count_parameters():,}",
            ")"
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the model
    print("Testing StockPredictionModel...")
    
    # Example configuration
    input_dim = 768 + 768 + 14 + 7  # stock_emb + news_emb + technical + ticker
    hidden_dims = [512, 256, 128, 64]
    
    model = StockPredictionModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=3,
        dropout=0.3,
        use_batch_norm=True,
        activation='leaky_relu'
    )
    
    print(model)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape (logits): {logits.shape}")
    
    proba = model.predict_proba(x)
    print(f"Output shape (proba): {proba.shape}")
    print(f"Probability sum (should be ~1.0): {proba[0].sum():.4f}")
    
    pred = model.predict(x)
    print(f"Output shape (predictions): {pred.shape}")
    print(f"Prediction range: {pred.min()}-{pred.max()}")
    
    print("\n✓ Model test passed!")

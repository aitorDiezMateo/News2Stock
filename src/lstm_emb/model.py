"""
LSTM Multi-Head Model Architecture
"""
import torch
import torch.nn as nn

class StockLSTMMultiHead(nn.Module):
    """
    LSTM model with multi-head architecture for embeddings generation
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(StockLSTMMultiHead, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # ===== SHARED ENCODER =====
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # ===== SPECIALIZED HEADS =====
        # Head 1: LOG_RETURN (momentum/trend)
        self.head1_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.head1_fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Head 2: ABS_LOG_RETURN (magnitude)
        self.head2_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.head2_fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Head 3: VOLATILITY (regime)
        self.head3_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.head3_fc2 = nn.Linear(hidden_size // 2, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x, return_embedding=False):
        # LSTM encoder
        lstm_out, _ = self.lstm(x)
        
        # Take last output as the embedding
        embedding = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Batch norm + dropout on embedding
        embedding_norm = self.batch_norm(embedding)
        embedding_dropped = self.dropout(embedding_norm)
        
        # Head 1
        h1 = self.relu(self.head1_fc1(embedding_dropped))
        out1 = self.head1_fc2(h1)
        
        # Head 2
        h2 = self.relu(self.head2_fc1(embedding_dropped))
        out2 = self.head2_fc2(h2)
        
        # Head 3
        h3 = self.relu(self.head3_fc1(embedding_dropped))
        out3 = self.head3_fc2(h3)
        
        # Concatenate outputs
        outputs = torch.cat([out1, out2, out3], dim=1)
        
        if return_embedding:
            return outputs, embedding
        return outputs
    
    def get_embedding(self, x):
        """Extract embedding without computing predictions"""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            embedding = lstm_out[:, -1, :]
            return embedding

class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss for multi-task learning"""
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(list(weights.values()), dtype=torch.float32)
        
    def forward(self, predictions, targets):
        if self.weights.device != predictions.device:
            self.weights = self.weights.to(predictions.device)
        
        mse_per_target = torch.mean((predictions - targets) ** 2, dim=0)
        weighted_loss = torch.sum(self.weights * mse_per_target)
        return weighted_loss

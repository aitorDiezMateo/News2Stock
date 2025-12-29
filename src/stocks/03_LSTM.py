"""
LSTM Model for Stock Prediction
================================

Predicts three targets:
  - LOG_RETURN: log returns
  - ABS_LOG_RETURN: absolute log returns
  - VOLATILITY: rolling volatility

Anti-overfitting techniques:
  - Dropout layers
  - L2 regularization
  - Early stopping
  - Batch normalization

Temporal splits:
  - Train: 2015-2021
  - Validation: 2021-2023
  - Test: 2024

Usage:
    python src/stocks/4_LSTM.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = 'data/stocks/processed/'
RESULTS_PATH = 'results/lstm/'
PLOTS_PATH = 'plots/lstm/'
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

TICKERS = ['GOOGL', 'AAPL', 'AMZN', 'META', 'MSFT', 'NVDA', 'TSLA']

# Model hyperparameters - OPTIMIZED FOR EMBEDDINGS & ANTI-OVERFITTING
SEQUENCE_LENGTH = 20  # Look back 20 days
HIDDEN_SIZE = 64      # Reduced from 128 to reduce overfitting
NUM_LAYERS = 2
DROPOUT = 0.5         # High dropout for regularization
L2_REG = 1e-3         # Increased L2 regularization (from 1e-4)

# Training hyperparameters
BATCH_SIZE = 128      # Increased batch size (from 64) for more stable gradients
LEARNING_RATE = 0.0005  # Reduced learning rate (from 0.001) for slower, more stable learning
EPOCHS = 200          # More epochs to compensate for slower learning
PATIENCE = 25         # Increased patience (from 15) to allow more exploration

# Weighted loss for balanced multi-task learning (for embeddings)
# Higher weights for harder-to-predict targets (returns) to force encoder to learn them
TARGET_WEIGHTS = {
    'LOG_RETURN': 3.0,        # Hardest - momentum/trend
    'ABS_LOG_RETURN': 2.0,    # Medium - magnitude
    'VOLATILITY': 1.0         # Easiest - regime
}

# Temporal splits
TRAIN_START = 2015
TRAIN_END = 2021
VAL_START = 2021
VAL_END = 2023
TEST_YEAR = 2024

# Target variables
TARGETS = ['LOG_RETURN', 'ABS_LOG_RETURN', 'VOLATILITY']

# Feature selection (updated based on available columns in processed data)
FEATURE_COLS = [
    # Price data
    'Close', 'High', 'Low', 'Open', 'Volume',
    # Moving averages
    'SMA_10', 'SMA_20', 'SMA_30',
    # Bollinger Bands
    'UPPER_BAND', 'MIDDLE_BAND', 'LOWER_BAND',
    # MACD
    'MACD', 'MACD_SIGNAL', 'MACD_HIST',
    # RSI
    'RSI_14',
    # Stochastic Oscillator
    'STOCH_K', 'STOCH_D',
    # Williams %R
    'WILLIAMS_R',
    # Log returns
    'LOG_RETURN_HIGH', 'LOG_RETURN_LOW', 'LOG_RETURN_OPEN', 'LOG_RETURN_CLOSE',
    # Volatility estimators
    'REALIZED_VOL', 'PARKINSON_VOL', 'GARMAN_KLASS_VOL', 'ROGERS_SATCHELL_VOL',
    # VWAP
    'VWAP',
    # Temporal features (cyclic encoded)
    'DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS',
    'MONTH_SIN', 'MONTH_COS',
    'DAY_OF_MONTH_SIN', 'DAY_OF_MONTH_COS',
    'QUARTER_SIN', 'QUARTER_COS'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_split_data(ticker):
    """Load data and split by temporal ranges"""
    print(f"\n{'='*80}")
    print(f"Loading {ticker}...")
    print('='*80)
    
    filepath = os.path.join(DATA_PATH, f"{ticker}_data_processed.parquet")
    df = pd.read_parquet(filepath)
    
    # Ensure Date column
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Add year column for splitting
    df['Year'] = df['Date'].dt.year
    
    # Check for required columns
    missing_features = [f for f in FEATURE_COLS if f not in df.columns]
    missing_targets = [t for t in TARGETS if t not in df.columns]
    
    if missing_features:
        print(f"  ⚠️  Missing features: {missing_features[:5]}...")
    if missing_targets:
        print(f"  ⚠️  Missing targets: {missing_targets}")
        return None, None, None, None, None, None
    
    # Select only available features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    
    # Temporal splits
    train_df = df[(df['Year'] >= TRAIN_START) & (df['Year'] <= TRAIN_END)].reset_index(drop=True)
    val_df = df[(df['Year'] >= VAL_START) & (df['Year'] <= VAL_END)].reset_index(drop=True)
    test_df = df[df['Year'] == TEST_YEAR].reset_index(drop=True)
    
    print(f"  Train: {len(train_df)} samples ({TRAIN_START}-{TRAIN_END})")
    print(f"  Val:   {len(val_df)} samples ({VAL_START}-{VAL_END})")
    print(f"  Test:  {len(test_df)} samples ({TEST_YEAR})")
    print(f"  Features: {len(available_features)}")
    
    return train_df, val_df, test_df, available_features, df['Date'], df

def create_sequences(data, features, targets, seq_length):
    """Create sequences for LSTM input"""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Input: seq_length days of features
        X.append(data[features].iloc[i:i+seq_length].values)
        # Target: next day's targets
        y.append(data[targets].iloc[i+seq_length].values)
    
    return np.array(X), np.array(y)

class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# LSTM MODEL
# ============================================================================

class StockLSTMMultiHead(nn.Module):
    """
    LSTM model with multi-head architecture for embeddings generation
    
    Architecture:
      [Input] → [LSTM Encoder] → [EMBEDDING] → [3 specialized heads]
      
    The encoder learns a shared representation (embedding) that captures:
      - Momentum/trend (LOG_RETURN)
      - Magnitude (ABS_LOG_RETURN)
      - Volatility regime (VOLATILITY)
    
    Anti-overfitting techniques:
      - Dropout between LSTM layers
      - Batch normalization
      - L2 regularization (in optimizer)
      - Early stopping (in training)
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(StockLSTMMultiHead, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # ===== SHARED ENCODER =====
        # LSTM encoder - learns the embedding representation
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization for the embedding
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # ===== SPECIALIZED HEADS =====
        # Each head specializes in one target
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
        """
        Forward pass
        
        Args:
            x: Input sequences [batch_size, seq_length, input_size]
            return_embedding: If True, returns (outputs, embedding) tuple
        
        Returns:
            outputs: Predictions [batch_size, 3]
            embedding (optional): Shared representation [batch_size, hidden_size]
        """
        # LSTM encoder
        lstm_out, _ = self.lstm(x)
        
        # Take last output as the embedding
        embedding = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Batch norm + dropout on embedding
        embedding_norm = self.batch_norm(embedding)
        embedding_dropped = self.dropout(embedding_norm)
        
        # Pass through each specialized head
        # Head 1: LOG_RETURN
        h1 = self.relu(self.head1_fc1(embedding_dropped))
        out1 = self.head1_fc2(h1)
        
        # Head 2: ABS_LOG_RETURN
        h2 = self.relu(self.head2_fc1(embedding_dropped))
        out2 = self.head2_fc2(h2)
        
        # Head 3: VOLATILITY
        h3 = self.relu(self.head3_fc1(embedding_dropped))
        out3 = self.head3_fc2(h3)
        
        # Concatenate outputs
        outputs = torch.cat([out1, out2, out3], dim=1)  # [batch_size, 3]
        
        if return_embedding:
            return outputs, embedding  # Return both predictions and embedding
        return outputs
    
    def get_embedding(self, x):
        """
        Extract embedding without computing predictions
        Useful for downstream tasks
        
        Args:
            x: Input sequences [batch_size, seq_length, input_size]
            
        Returns:
            embedding: [batch_size, hidden_size]
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            embedding = lstm_out[:, -1, :]
            return embedding

# ============================================================================
# TRAINING
# ============================================================================

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for multi-task learning
    
    Applies different weights to each target to balance their contribution
    to the total loss. This is crucial for embeddings learning as it forces
    the encoder to learn features useful for all targets, not just the easiest one.
    """
    def __init__(self, weights):
        """
        Args:
            weights: dict with target names as keys and weights as values
                     e.g., {'LOG_RETURN': 3.0, 'ABS_LOG_RETURN': 2.0, 'VOLATILITY': 1.0}
        """
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(list(weights.values()), dtype=torch.float32)
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, num_targets]
            targets: [batch_size, num_targets]
        Returns:
            weighted_loss: scalar
        """
        # Move weights to same device as predictions
        if self.weights.device != predictions.device:
            self.weights = self.weights.to(predictions.device)
        
        # Compute MSE for each target
        mse_per_target = torch.mean((predictions - targets) ** 2, dim=0)  # [num_targets]
        
        # Apply weights
        weighted_loss = torch.sum(self.weights * mse_per_target)
        
        return weighted_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
    """Train with early stopping and learning rate scheduling"""
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Learning rate scheduler - reduces LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    print("\n" + "="*80)
    print("Training started...")
    print("="*80)
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (additional anti-overfitting)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            model.load_state_dict(best_model_state)
            break
    
    return model, train_losses, val_losses

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, loader, dataset_name):
    """Evaluate model and compute metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Compute metrics for each target
    metrics = {}
    for i, target_name in enumerate(TARGETS):
        mse = mean_squared_error(targets[:, i], preds[:, i])
        mae = mean_absolute_error(targets[:, i], preds[:, i])
        r2 = r2_score(targets[:, i], preds[:, i])
        
        metrics[target_name] = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R2': r2
        }
    
    # Print results
    print(f"\n📊 {dataset_name} Results:")
    print("-" * 80)
    for target_name, target_metrics in metrics.items():
        print(f"  {target_name}:")
        for metric_name, value in target_metrics.items():
            print(f"    {metric_name}: {value:.6f}")
    
    return metrics, preds, targets

def plot_predictions(preds, targets, ticker, dataset_name, dates_subset=None):
    """Plot predictions vs actual for each target"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for i, target_name in enumerate(TARGETS):
        ax = axes[i]
        x = dates_subset if dates_subset is not None else range(len(targets))
        
        ax.plot(x, targets[:, i], label='Actual', alpha=0.7)
        ax.plot(x, preds[:, i], label='Predicted', alpha=0.7)
        ax.set_title(f'{ticker} - {target_name} ({dataset_name})')
        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time' if dates_subset is None else 'Date')
    fig.tight_layout()
    
    plot_path = os.path.join(PLOTS_PATH, f"{ticker}_{dataset_name}_predictions.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  📈 Saved plot: {plot_path}")
    plt.close(fig)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_lstm_for_ticker(ticker):
    """Complete pipeline for one ticker with target normalization"""
    print(f"\n{'#'*80}")
    print(f"# Processing {ticker}")
    print('#'*80)
    
    # Load data
    train_df, val_df, test_df, features, dates, full_df = load_and_split_data(ticker)
    if train_df is None:
        print(f"  ❌ Skipping {ticker} due to missing data")
        return None
    
    # ===== SCALE FEATURES =====
    print("\n📊 Scaling features...")
    feature_scaler = RobustScaler()
    train_df[features] = feature_scaler.fit_transform(train_df[features])
    val_df[features] = feature_scaler.transform(val_df[features])
    test_df[features] = feature_scaler.transform(test_df[features])
    
    # ===== SCALE TARGETS INDIVIDUALLY (crucial for balanced embeddings!) =====
    print("📊 Scaling targets individually...")
    target_scalers = {}
    for target in TARGETS:
        scaler = RobustScaler()
        # Fit on train, transform all
        train_df[[target]] = scaler.fit_transform(train_df[[target]])
        val_df[[target]] = scaler.transform(val_df[[target]])
        test_df[[target]] = scaler.transform(test_df[[target]])
        target_scalers[target] = scaler
        print(f"  {target}: median={scaler.center_[0]:.6f}, scale={scaler.scale_[0]:.6f}")
    
    # Create sequences
    print("\nCreating sequences...")
    X_train, y_train = create_sequences(train_df, features, TARGETS, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_df, features, TARGETS, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_df, features, TARGETS, SEQUENCE_LENGTH)
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    # Create DataLoaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model with multi-head architecture
    input_size = len(features)
    output_size = len(TARGETS)
    
    model = StockLSTMMultiHead(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗️  Model: {total_params:,} parameters")
    print(f"  Architecture: Multi-Head LSTM for Embeddings")
    print(f"  Encoder: {input_size} → LSTM({HIDDEN_SIZE}) → Embedding({HIDDEN_SIZE})")
    print(f"  Heads: 3 specialized heads (one per target)")
    
    # Weighted loss and optimizer (with L2 regularization)
    criterion = WeightedMSELoss(TARGET_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
    
    print(f"\n⚖️  Loss weights: {TARGET_WEIGHTS}")
    
    # Train
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS, PATIENCE
    )
    
    # Evaluate on all sets (predictions will be in normalized space)
    train_metrics, train_preds, train_targets = evaluate_model(model, train_loader, "TRAIN")
    val_metrics, val_preds, val_targets = evaluate_model(model, val_loader, "VALIDATION")
    test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, "TEST")
    
    # Denormalize predictions and targets for interpretable metrics
    print("\n🔄 Denormalizing predictions for final metrics...")
    for i, target in enumerate(TARGETS):
        scaler = target_scalers[target]
        # Denormalize
        train_preds[:, i] = scaler.inverse_transform(train_preds[:, i].reshape(-1, 1)).flatten()
        train_targets[:, i] = scaler.inverse_transform(train_targets[:, i].reshape(-1, 1)).flatten()
        
        val_preds[:, i] = scaler.inverse_transform(val_preds[:, i].reshape(-1, 1)).flatten()
        val_targets[:, i] = scaler.inverse_transform(val_targets[:, i].reshape(-1, 1)).flatten()
        
        test_preds[:, i] = scaler.inverse_transform(test_preds[:, i].reshape(-1, 1)).flatten()
        test_targets[:, i] = scaler.inverse_transform(test_targets[:, i].reshape(-1, 1)).flatten()
    
    # Recompute metrics on denormalized data
    print("\n📊 FINAL METRICS (Denormalized):")
    train_metrics_final = {}
    val_metrics_final = {}
    test_metrics_final = {}
    
    for i, target in enumerate(TARGETS):
        for metrics_dict, preds, targets, name in [
            (train_metrics_final, train_preds, train_targets, "TRAIN"),
            (val_metrics_final, val_preds, val_targets, "VALIDATION"),
            (test_metrics_final, test_preds, test_targets, "TEST")
        ]:
            mse = mean_squared_error(targets[:, i], preds[:, i])
            mae = mean_absolute_error(targets[:, i], preds[:, i])
            r2 = r2_score(targets[:, i], preds[:, i])
            
            metrics_dict[target] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mae,
                'R2': r2
            }
    
    # Print final metrics
    for name, metrics_dict in [("TRAIN", train_metrics_final), ("VALIDATION", val_metrics_final), ("TEST", test_metrics_final)]:
        print(f"\n  {name}:")
        for target, target_metrics in metrics_dict.items():
            print(f"    {target}: R²={target_metrics['R2']:.4f}, RMSE={target_metrics['RMSE']:.6f}")
    
    # Get dates for plotting
    train_dates = train_df['Date'].iloc[SEQUENCE_LENGTH:].values
    val_dates = val_df['Date'].iloc[SEQUENCE_LENGTH:].values
    test_dates = test_df['Date'].iloc[SEQUENCE_LENGTH:].values
    
    # Plot predictions
    plot_predictions(train_preds, train_targets, ticker, "train", train_dates)
    plot_predictions(val_preds, val_targets, ticker, "validation", val_dates)
    plot_predictions(test_preds, test_targets, ticker, "test", test_dates)
    
    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted Loss')
    ax.set_title(f'{ticker} - Training Curves (Weighted Multi-Task Loss)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    curve_path = os.path.join(PLOTS_PATH, f"{ticker}_training_curves.png")
    fig.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"  📉 Saved training curves: {curve_path}")
    plt.close(fig)
    
    # Save metrics to CSV (use denormalized metrics)
    results = []
    for dataset_name, metrics in [('train', train_metrics_final), ('validation', val_metrics_final), ('test', test_metrics_final)]:
        for target_name, target_metrics in metrics.items():
            row = {
                'ticker': ticker,
                'dataset': dataset_name,
                'target': target_name,
                **target_metrics
            }
            results.append(row)
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_PATH, f"{ticker}_metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  💾 Saved metrics: {results_path}")
    
    # Save model with scalers
    model_path = os.path.join(RESULTS_PATH, f"{ticker}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'features': features,
        'feature_scaler': feature_scaler,
        'target_scalers': target_scalers,
        'target_weights': TARGET_WEIGHTS,
        'config': {
            'input_size': input_size,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'output_size': output_size,
            'dropout': DROPOUT,
            'architecture': 'StockLSTMMultiHead'
        }
    }, model_path)
    print(f"  💾 Saved model: {model_path}")
    
    # ===== EXTRACT AND SAVE EMBEDDINGS =====
    print("\n🔮 Extracting embeddings...")
    embeddings_path = 'data/embeddings/lstm_multihead/'
    os.makedirs(embeddings_path, exist_ok=True)
    
    # Extract embeddings for all splits
    train_embeddings, _ = extract_embeddings(model, train_loader, DEVICE)
    val_embeddings, _ = extract_embeddings(model, val_loader, DEVICE)
    test_embeddings, _ = extract_embeddings(model, test_loader, DEVICE)
    
    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Val embeddings: {val_embeddings.shape}") 
    print(f"  Test embeddings: {test_embeddings.shape}")
    
    # Save embeddings
    train_emb_path = os.path.join(embeddings_path, f"{ticker}_train_embeddings.npz")
    np.savez_compressed(train_emb_path, embeddings=train_embeddings, targets=train_targets)
    print(f"  💾 Saved: {train_emb_path}")
    
    val_emb_path = os.path.join(embeddings_path, f"{ticker}_val_embeddings.npz")
    np.savez_compressed(val_emb_path, embeddings=val_embeddings, targets=val_targets)
    print(f"  💾 Saved: {val_emb_path}")
    
    test_emb_path = os.path.join(embeddings_path, f"{ticker}_test_embeddings.npz")
    np.savez_compressed(test_emb_path, embeddings=test_embeddings, targets=test_targets)
    print(f"  💾 Saved: {test_emb_path}")
    
    print(f"\n✅ {ticker} processing complete!")
    
    return results_df

# ============================================================================
# EMBEDDING EXTRACTION UTILITIES
# ============================================================================

def extract_embeddings(model, dataloader, device=DEVICE):
    """
    Extract embeddings from the trained model
    
    Args:
        model: Trained StockLSTMMultiHead model
        dataloader: DataLoader with sequences
        device: torch device
        
    Returns:
        embeddings: numpy array [num_samples, hidden_size]
        targets: numpy array [num_samples, num_targets] (optional, for analysis)
    """
    model.eval()
    all_embeddings = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            
            # Extract embeddings
            embeddings = model.get_embedding(X_batch)  # [batch_size, hidden_size]
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    embeddings = np.vstack(all_embeddings)
    targets = np.vstack(all_targets)
    
    return embeddings, targets


def save_embeddings(ticker, embeddings_dict, save_path='data/embeddings/'):
    """
    Save embeddings for a ticker
    
    Args:
        ticker: Stock ticker
        embeddings_dict: Dict with 'train', 'val', 'test' keys containing embeddings
        save_path: Directory to save embeddings
    """
    os.makedirs(save_path, exist_ok=True)
    
    for split_name, (embeddings, targets) in embeddings_dict.items():
        filename = os.path.join(save_path, f"{ticker}_{split_name}_embeddings.npz")
        np.savez_compressed(
            filename,
            embeddings=embeddings,
            targets=targets
        )
        print(f"  💾 Saved {split_name} embeddings: {filename} (shape: {embeddings.shape})")


def load_trained_model_and_extract_embeddings(ticker, model_path=None):
    """
    Load a trained model and extract embeddings for all splits
    
    Args:
        ticker: Stock ticker
        model_path: Path to saved model (if None, uses default)
        
    Returns:
        embeddings_dict: Dict with train/val/test embeddings
    """
    if model_path is None:
        model_path = os.path.join(RESULTS_PATH, f"{ticker}_model.pt")
    
    print(f"\n{'='*80}")
    print(f"Extracting embeddings for {ticker}")
    print('='*80)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint['config']
    
    # Recreate model
    model = StockLSTMMultiHead(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✅ Model loaded from {model_path}")
    
    # Load and prepare data
    train_df, val_df, test_df, features, _, _ = load_and_split_data(ticker)
    
    # Scale features
    feature_scaler = checkpoint['feature_scaler']
    train_df[features] = feature_scaler.transform(train_df[features])
    val_df[features] = feature_scaler.transform(val_df[features])
    test_df[features] = feature_scaler.transform(test_df[features])
    
    # Create sequences
    X_train, y_train = create_sequences(train_df, features, TARGETS, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_df, features, TARGETS, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_df, features, TARGETS, SEQUENCE_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # Extract embeddings
    print("\n  Extracting embeddings...")
    train_emb, train_tgt = extract_embeddings(model, train_loader)
    val_emb, val_tgt = extract_embeddings(model, val_loader)
    test_emb, test_tgt = extract_embeddings(model, test_loader)
    
    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Val embeddings: {val_emb.shape}")
    print(f"  Test embeddings: {test_emb.shape}")
    
    embeddings_dict = {
        'train': (train_emb, train_tgt),
        'val': (val_emb, val_tgt),
        'test': (test_emb, test_tgt)
    }
    
    return embeddings_dict

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("LSTM Multi-Head Stock Embedding & Prediction Pipeline")
    print("="*80)
    print(f"Targets: {TARGETS}")
    print(f"Target Weights: {TARGET_WEIGHTS}")
    print(f"Architecture: Multi-Head LSTM for Embeddings")
    print(f"Train: {TRAIN_START}-{TRAIN_END}")
    print(f"Val: {VAL_START}-{VAL_END}")
    print(f"Test: {TEST_YEAR}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"Device: {DEVICE}")
    
    all_results = []
    
    for ticker in TICKERS:
        try:
            results = run_lstm_for_ticker(ticker)
            if results is not None:
                all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(RESULTS_PATH, "all_tickers_metrics.csv")
        combined_results.to_csv(combined_path, index=False)
        print(f"\n✅ Combined metrics saved: {combined_path}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("Summary Statistics (Test Set)")
        print("="*80)
        test_results = combined_results[combined_results['dataset'] == 'test']
        summary = test_results.groupby('target')[['MSE', 'RMSE', 'MAE', 'R2']].mean()
        print(summary)
    
    print("\n" + "="*80)
    print("✅ Pipeline completed!")
    print("="*80)

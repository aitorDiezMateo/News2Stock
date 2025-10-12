import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Config
DATA_PATH = 'data_stocks/processed'
STOCKS = ['GOOGL', 'AMZN', 'AAPL', 'META', 'MSFT', 'NVDA', 'TSLA']
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDINGS_DIR = 'embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Dataset (multivariate)
class StockDataset(Dataset):
    def __init__(self, features, target, seq_len):
        """features: array shape (N, n_features)
           target: array shape (N,)
        """
        self.seq_len = seq_len
        self.features = features
        self.target = target
        self.X, self.y = self.create_sequences(features, target, seq_len)

    def create_sequences(self, features, target, seq_len):
        xs, ys = [], []
        for i in range(len(features) - seq_len):
            xs.append(features[i:i+seq_len])
            ys.append(target[i+seq_len])
        return np.array(xs), np.array(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        # Save the last hidden state as embedding
        embedding = hn[-1]
        out = self.fc(out[:, -1, :])
        return out, embedding

for stock in STOCKS:
    print(f"Processing {stock}...")
    df = pd.read_parquet(f"{DATA_PATH}/{stock}_data_processed.parquet")
    # Use all numeric columns as features, and 'Close' as the target
    numeric = df.select_dtypes(include=[np.number])
    if 'Close' not in numeric.columns:
        raise ValueError(f"Data for {stock} does not contain 'Close' column")

    features_df = numeric.drop(columns=['Close']).ffill().bfill()
    target_df = numeric['Close'].ffill().bfill()

    # scale features and target separately
    feat_scaler = MinMaxScaler()
    targ_scaler = MinMaxScaler()

    features = feat_scaler.fit_transform(features_df.values)
    target = targ_scaler.fit_transform(target_df.values.reshape(-1, 1)).flatten()

    n_features = features.shape[1]

    dataset = StockDataset(features, target, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(input_size=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for X_batch, y_batch in dataloader:
            # X_batch already has shape (batch, seq_len, n_features)
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"{stock} Epoch {epoch+1}/{EPOCHS} Loss: {np.mean(losses):.4f}")

    # Save embeddings for the whole series
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(len(features) - SEQ_LEN):
            seq = torch.FloatTensor(features[i:i+SEQ_LEN]).unsqueeze(0).to(DEVICE)  # shape (1, seq_len, n_features)
            _, embedding = model(seq)
            all_embeddings.append(embedding.cpu().numpy().flatten())
    all_embeddings = np.array(all_embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, f"{stock}_embeddings.npy"), all_embeddings)
    print(f"Saved embeddings for {stock} to {EMBEDDINGS_DIR}/{stock}_embeddings.npy")
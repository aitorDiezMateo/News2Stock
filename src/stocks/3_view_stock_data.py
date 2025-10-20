import matplotlib.pyplot as plt
import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = 'plots/stock'
os.makedirs(output_dir, exist_ok=True)

# Load data
data_google = pd.read_parquet('data/stocks/processed/GOOGL_data_processed.parquet')
data_amazon = pd.read_parquet('data/stocks/processed/AMZN_data_processed.parquet')
data_apple = pd.read_parquet('data/stocks/processed/AAPL_data_processed.parquet')
data_meta = pd.read_parquet('data/stocks/processed/META_data_processed.parquet')
data_microsoft = pd.read_parquet('data/stocks/processed/MSFT_data_processed.parquet')
data_nvidia = pd.read_parquet('data/stocks/processed/NVDA_data_processed.parquet')
data_tesla = pd.read_parquet('data/stocks/processed/TSLA_data_processed.parquet')

# Prepare data for plotting
stock_data = {
    'GOOGL': data_google,
    'AMZN': data_amazon,
    'AAPL': data_apple,
    'META': data_meta,
    'MSFT': data_microsoft,
    'NVDA': data_nvidia,
    'TSLA': data_tesla
}


import math

# Helper to create subplots grid
def _grid_shape(n):
    cols = 2
    rows = math.ceil(n / cols)
    return rows, cols

# Plot: each ticker in its own subplot (same figure)
def plot_subplots(stock_dict, value_col, fig_name, ylabel, title_prefix):
    n = len(stock_dict)
    rows, cols = _grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=False)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax_idx, (ticker, df) in enumerate(stock_dict.items()):
        ax = axes[ax_idx]
        if value_col not in df.columns:
            ax.text(0.5, 0.5, f'{value_col} not found', ha='center')
        else:
            ax.plot(df['Date'] if 'Date' in df.columns else df.index, df[value_col])
        ax.set_title(ticker)
        ax.set_ylabel(ylabel)

    # Turn off unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'{title_prefix} (each subplot = one ticker)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, fig_name)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


# Create and save subplot figures for prices and log returns
plot_subplots(stock_data, 'Close', 'stock_prices_subplots.png', 'Price', 'Stock Prices Over Time')
plot_subplots(stock_data, 'LOG_RETURN', 'stock_log_returns_subplots.png', 'Log Return', 'Log Returns Over Time')
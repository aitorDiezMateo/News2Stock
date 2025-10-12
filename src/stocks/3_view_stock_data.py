import matplotlib.pyplot as plt
import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = 'plots/stock'
os.makedirs(output_dir, exist_ok=True)

# Load data
data_google = pd.read_parquet('data_stocks/processed/GOOGL_data_processed.parquet')
data_amazon = pd.read_parquet('data_stocks/processed/AMZN_data_processed.parquet')
data_apple = pd.read_parquet('data_stocks/processed/AAPL_data_processed.parquet')
data_meta = pd.read_parquet('data_stocks/processed/META_data_processed.parquet')
data_microsoft = pd.read_parquet('data_stocks/processed/MSFT_data_processed.parquet')
data_nvidia = pd.read_parquet('data_stocks/processed/NVDA_data_processed.parquet')
data_tesla = pd.read_parquet('data_stocks/processed/TSLA_data_processed.parquet')

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


# Plot all stocks in the same figure
plt.figure(figsize=(12, 8))
for ticker, df in stock_data.items():
    # Try to use 'Close' price, fallback to first numeric column if not present
    price_col = 'Close' if 'Close' in df.columns else df.select_dtypes('number').columns[0]
    plt.plot(df.index, df[price_col], label=ticker)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices Over Time')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'all_stocks.png'))
plt.close()

n = len(stock_data)
fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

if n == 1:
    axes = [axes]

for ax, (ticker, df) in zip(axes, stock_data.items()):
    price_col = 'Close'
    ax.plot(df.index, df[price_col], label=f'{ticker} Price')
    ax.plot(df.index, df['SMA_30'], label=f'{ticker} SMA_30', linestyle='--')
    ax.set_title(f'{ticker} Price and 30-Day SMA')
    ax.set_ylabel('Price')
    ax.legend()

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stocks_with_moving_average.png'))
plt.close()


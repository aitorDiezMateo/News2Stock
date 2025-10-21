import pandas as pd
from pathlib import Path

# Define paths
processed_dir = Path(__file__).parent.parent / "data" / "news" / "processed"
summarized_dir = Path(__file__).parent.parent / "data" / "news" / "summarized"

# Load all processed news files
processed_files = list(processed_dir.glob("*.parquet"))
df_processed = pd.concat(
    [pd.read_parquet(file) for file in processed_files],
    ignore_index=True
)

# Load all summarized news files
summarized_files = list(summarized_dir.glob("*.parquet"))
df_summarized = pd.concat(
    [pd.read_parquet(file) for file in summarized_files],
    ignore_index=True
)

print(f"Loaded {len(df_processed)} rows from processed data")
print(f"Loaded {len(df_summarized)} rows from summarized data")
print(f"\nProcessed columns: {df_processed.columns.tolist()}")
print(f"\nSummarized columns: {df_summarized.columns.tolist()}")

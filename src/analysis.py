import pandas as pd
from pathlib import Path

import re
# Define paths
processed_dir = Path(__file__).parent.parent / "data" / "news" / "processed"
summarized_dir = Path(__file__).parent.parent / "data" / "news" / "summarized"

# # Load all processed news files
# processed_files = list(processed_dir.glob("*.parquet"))
# df_processed = pd.concat(
#     [pd.read_parquet(file) for file in processed_files],
#     ignore_index=True
# )

# # Load all summarized news files
# summarized_files = list(summarized_dir.glob("*.parquet"))
# df_summarized = pd.concat(
#     [pd.read_parquet(file) for file in summarized_files],
#     ignore_index=True
# )


# print(f"Loaded {len(df_processed)} rows from processed data")
# print(f"Loaded {len(df_summarized)} rows from summarized data")
# print(f"\nProcessed columns: {df_processed.columns.tolist()}")
# print(f"\nSummarized columns: {df_summarized.columns.tolist()}")

df_summarized = pd.read_parquet(summarized_dir / "amazon_news.parquet")

df_summarized["text"] = df_summarized["title"] + "\n\n" + df_summarized["body_summary"]


# Buscar filas donde la palabra 'salvagedata' aparezca 3 veces
mask = df_summarized['text'].str.count(r'\bsalvagedata\b', flags=re.IGNORECASE) == 3

# Obtener la primera fila que cumple la condición
fila = df_summarized[mask].head(1)

# Imprimirla
print(fila["body_summary"].values[0])
print(fila["clean_body"].values[0])

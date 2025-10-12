import pandas as pd
from bs4 import BeautifulSoup
import os
import re
import html

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'raw')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Listar todos los archivos parquet en el directorio RAW_DIR
parquet_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.parquet')]

print(f"Found {len(parquet_files)} parquet files to process")

# Procesar cada archivo
for filename in parquet_files:
    print(f"\nProcessing: {filename}")
    
    input_path = os.path.join(RAW_DIR, filename)
    df = pd.read_parquet(input_path)
    
    # Eliminar las filas con valores nulos en la columna body y teaser
    df = df.dropna(subset=["body", "teaser"], how="all").reset_index(drop=True)
    
    # Limpiar HTML y extraer texto
    df["clean_body"] = df["body"].apply(
        lambda x: BeautifulSoup(x, "html.parser").get_text(separator="\n", strip=True)
    )
    
    # Convertir entidades HTML a caracteres normales
    df["clean_body"] = df["clean_body"].apply(html.unescape)
    
    # Reemplazar saltos de línea pegados a palabras por un espacio
    df["clean_body"] = df["clean_body"].apply(lambda x: re.sub(r'(?<=\S)\n(?=\S)', ' ', x))
    
    # Reducir saltos de línea múltiples a uno solo
    df["clean_body"] = df["clean_body"].apply(lambda x: re.sub(r'\n+', '\n', x))
    
    # Limpiar espacios sobrantes al inicio/final
    df["clean_body"] = df["clean_body"].str.strip()
    
    # Eliminar la columna original body
    df = df.drop(columns=["body"])
    
    # Guardar con el mismo nombre de archivo
    output_path = os.path.join(OUTPUT_DIR, filename)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved: {output_path} ({len(df)} rows)")

print("\nAll files processed!")
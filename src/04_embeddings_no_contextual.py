import fasttext.util
import pandas as pd
import os
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'summarized')
FASTTEXT_PATH = os.path.join(ROOT_DIR, 'models', 'cc.en.300.bin')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'embeddings')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load FastText model
ft = fasttext.load_model(FASTTEXT_PATH)
print("FastText model loaded.")

# Generate embeddings for each text
def get_text_embedding(tokens):
    embeddings = []
    for token in tokens:
        embeddings.append(ft.get_word_vector(token))
    
    if len(embeddings) == 0:
        return np.zeros(ft.get_dimension())
    
    return np.mean(embeddings, axis=0)

# Get all parquet files in the summarized directory
parquet_files = [f for f in os.listdir(SUMMARIES_DIR) if f.endswith('.parquet')]

print(f"Found {len(parquet_files)} files to process.\n")

# Process each file
for file_name in parquet_files:
    print(f"Processing {file_name}...")
    
    # Load dataset
    df = pd.read_parquet(os.path.join(SUMMARIES_DIR, file_name), engine="pyarrow")
    
    df["text"] = df["title"] + "\n\n" + df["body_summary"]
    df = df.drop(columns=["clean_body", "body_summary", "teaser", "title", "author"])
    
    # Remove rows with missing text
    df = df.dropna(subset=['text'])
    
    # Tokenize the 'text' column
    df['tokens'] = df['text'].apply(lambda x: word_tokenize(x.lower()))
    
    # Generate embeddings
    df['embedding'] = df['tokens'].apply(get_text_embedding)
    
    # Drop tokens column
    df = df.drop(columns=['tokens'])
    
    # Save to output directory with modified filename
    output_name = file_name.replace('_news.parquet', '_embeddings_no_context.parquet')
    output_path = os.path.join(OUTPUT_DIR, output_name)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    
    print(f"  Saved {output_name} with {len(df)} records and embedding dimension {ft.get_dimension()}\n")

print("All files processed successfully!")
# Embedding DIM: 300
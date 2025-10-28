from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'summarized')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'embeddings')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load FinBERT model and tokenizer
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert")
model.eval()
print("FinBERT model loaded.")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Generate embeddings for each text
def get_text_embedding(text):
    """
    Get the contextual embedding for a text using FinBERT.
    Returns the [CLS] token embedding as the text representation.
    """
    # Tokenize and prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding (first token) as text representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return cls_embedding.cpu().numpy()

# Get all parquet files in the summarized directory
parquet_files = [f for f in os.listdir(SUMMARIES_DIR) if f.endswith('.parquet')]

print(f"Found {len(parquet_files)} files to process.\n")

# Process each file
for file_idx, file_name in enumerate(parquet_files, 1):
    print(f"[{file_idx}/{len(parquet_files)}] Processing {file_name}...")
    
    # Load dataset
    df = pd.read_parquet(os.path.join(SUMMARIES_DIR, file_name), engine="pyarrow")
    
    df["text"] = df["title"] + "\n\n" + df["body_summary"]
    df = df.drop(columns=["clean_body", "body_summary", "teaser", "title", "author"])
    
    # Remove rows with missing text
    original_count = len(df)
    df = df.dropna(subset=['text'])
    final_count = len(df)
    
    if original_count != final_count:
        print(f"  Removed {original_count - final_count} rows with missing text")
    
    print(f"  Generating embeddings for {final_count} texts...")
    
    # Generate embeddings with progress tracking
    embeddings = []
    for idx, text in enumerate(df['text'], 1):
        if idx % 100 == 0 or idx == final_count:
            print(f"    Progress: {idx}/{final_count} ({100*idx/final_count:.1f}%)", end='\r')
        embeddings.append(get_text_embedding(text))
    
    df['embedding'] = embeddings
    print()  # New line after progress
    
    # Save to output directory with modified filename
    output_name = file_name.replace('_news.parquet', '_embeddings_contextual.parquet')
    output_path = os.path.join(OUTPUT_DIR, output_name)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    
    embedding_dim = len(df['embedding'].iloc[0])
    print(f"  ✓ Saved {output_name} with {len(df)} records and embedding dimension {embedding_dim}\n")

print("All files processed successfully!")

# EMBEDDING DIM: 768
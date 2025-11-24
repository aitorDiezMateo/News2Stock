"""
Data preprocessing script for Pointer-Generator Network
"""

import pandas as pd
import re
import pickle
from typing import List, Tuple
import config
from vocabulary import Vocabulary


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Simple word tokenization"""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def load_and_preprocess_data(data_path: str = None) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load and preprocess the news dataset
    
    Args:
        data_path: Path to a single parquet file, or None to load all companies
    
    Returns:
        sources: List of tokenized source texts
        targets: List of tokenized target summaries
    """
    if data_path is None:
        # Load all company data
        companies = ['amazon', 'apple', 'google', 'meta', 'microsoft', 'nvidia', 'tesla']
        data_dir = config.ROOT_DIR + '/data/news/summarized/'
        
        print("Loading data from all companies...")
        dfs = []
        for company in companies:
            company_path = f"{data_dir}{company}_news.parquet"
            try:
                df_company = pd.read_parquet(company_path)
                print(f"  ✓ Loaded {company}: {len(df_company)} examples")
                dfs.append(df_company)
            except FileNotFoundError:
                print(f"  ✗ Skipped {company}: file not found")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Total combined: {len(df)} examples")
    else:
        # Load single file
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Use specific columns for news dataset
    source_col = 'clean_body'
    target_col = 'body_summary'
    
    # Verify columns exist
    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    print(f"\n✓ Using '{source_col}' as source (original text)")
    print(f"✓ Using '{target_col}' as target (summary)")
    
    # Filter out rows with missing data
    df = df[[source_col, target_col]].dropna()
    print(f"\nAfter removing NaN: {len(df)} examples")
    
    # Clean and tokenize
    print("\nCleaning and tokenizing...")
    sources = []
    targets = []
    
    for idx, row in df.iterrows():
        source_text = clean_text(row[source_col])
        target_text = clean_text(row[target_col])
        
        if len(source_text) > 0 and len(target_text) > 0:
            source_tokens = tokenize(source_text)
            target_tokens = tokenize(target_text)
            
            # Filter out very short examples
            if len(source_tokens) >= 10 and len(target_tokens) >= 3:
                sources.append(source_tokens)
                targets.append(target_tokens)
    
    print(f"Final dataset size: {len(sources)} examples")
    
    # Print statistics
    source_lens = [len(s) for s in sources]
    target_lens = [len(t) for t in targets]
    
    print(f"\nSource statistics:")
    print(f"  Mean length: {sum(source_lens) / len(source_lens):.1f}")
    print(f"  Max length: {max(source_lens)}")
    print(f"  Min length: {min(source_lens)}")
    
    print(f"\nTarget statistics:")
    print(f"  Mean length: {sum(target_lens) / len(target_lens):.1f}")
    print(f"  Max length: {max(target_lens)}")
    print(f"  Min length: {min(target_lens)}")
    
    return sources, targets


def split_data(sources: List[List[str]], targets: List[List[str]], 
               train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Split data into train, validation, and test sets
    
    Args:
        sources: List of tokenized source texts
        targets: List of tokenized target summaries
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (rest goes to test)
    """
    n = len(sources)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_sources = sources[:train_size]
    train_targets = targets[:train_size]
    
    val_sources = sources[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    
    test_sources = sources[train_size + val_size:]
    test_targets = targets[train_size + val_size:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_sources)} examples")
    print(f"  Val: {len(val_sources)} examples")
    print(f"  Test: {len(test_sources)} examples")
    
    return (train_sources, train_targets), (val_sources, val_targets), (test_sources, test_targets)



print("="*80)
print("PREPROCESSING DATA FOR POINTER-GENERATOR NETWORK")
print("="*80)

# Load and preprocess data from all companies
# Set data_path=None to load all companies, or specify a path for single company
sources, targets = load_and_preprocess_data(data_path=None)  # Load all companies

# Split data
train_data, val_data, test_data = split_data(sources, targets)

# Build vocabulary from training data only
print("\n" + "="*80)
print("BUILDING VOCABULARY")
print("="*80)
vocab = Vocabulary()

# Combine train sources and targets for vocabulary
all_train_texts = train_data[0] + train_data[1]
vocab.build_vocabulary(all_train_texts, min_freq=config.MIN_WORD_FREQ, max_size=config.VOCAB_SIZE)

# Save preprocessed data
print("\n" + "="*80)
print("SAVING PREPROCESSED DATA")
print("="*80)

data_dict = {
    'train': train_data,
    'val': val_data,
    'test': test_data,
    'vocab': vocab
}

output_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Saved preprocessed data to: {output_path}")
print(f"Vocabulary size: {len(vocab)}")
print("\nPreprocessing complete!")

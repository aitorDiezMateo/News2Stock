# ==========================
# IMPORTS
# ==========================
import pandas as pd
import os
import string
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'summarized')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'traditional')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# LOAD SPACY MODEL
# ==========================
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_lg") # to install it: python -m spacy download en_core_web_lg
# It does:
    # Tokens
    # Lemmatization
    # Stopword removal
    # Grammatical features
    # Syntactic dependencies
print("spaCy model loaded.")

# ==========================
# PREPROCESSING FUNCTIONS
# ==========================

def clean_text(text):
    """
    Clean text: lowercasing, remove punctuation and numbers.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([c for c in text if not c.isdigit()])
    return text

def preprocess_text(text):
    """
    Clean and lemmatize text, removing stopwords.
    """
    text = clean_text(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.lemma_ != '-PRON-']
    return " ".join(tokens)

# Get all parquet files in the summarized directory
parquet_files = [f for f in os.listdir(SUMMARIES_DIR) if f.endswith('.parquet')]

print(f"Found {len(parquet_files)} files to process.\n")

# Batch size for processing
BATCH_SIZE = 500

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
    
    print(f"  Preprocessing {final_count} texts in batches of {BATCH_SIZE}...")
    
    # Apply preprocessing in batches
    processed_texts = []
    num_batches = (final_count + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, final_count)
        
        batch_texts = df['text'].iloc[start_idx:end_idx]
        
        for idx, text in enumerate(batch_texts, start=start_idx + 1):
            if idx % 100 == 0 or idx == final_count:
                print(f"    Progress: {idx}/{final_count} ({100*idx/final_count:.1f}%)", end='\r')
            processed_texts.append(preprocess_text(text))
    
    df['text_processed'] = processed_texts
    print()  # New line after progress
    
    # ==========================
    # BAG OF WORDS (in batches)
    # ==========================
    print("  Generating Bag of Words representation...")
    bow_vectorizer = CountVectorizer(max_features=5000)  # Limit features to reduce memory
    
    # Fit vectorizer on all data
    bow_vectorizer.fit(df['text_processed'])
    
    # Transform in batches
    bow_vectors = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, final_count)
        
        batch_processed = df['text_processed'].iloc[start_idx:end_idx]
        batch_bow = bow_vectorizer.transform(batch_processed)
        bow_vectors.extend(batch_bow.toarray())
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            print(f"    Batch {batch_idx + 1}/{num_batches} processed", end='\r')
    
    df['bow'] = bow_vectors
    print()  # New line
    
    # ==========================
    # TF-IDF (in batches)
    # ==========================
    print("  Generating TF-IDF representation...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to reduce memory
    
    # Fit vectorizer on all data
    tfidf_vectorizer.fit(df['text_processed'])
    
    # Transform in batches
    tfidf_vectors = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, final_count)
        
        batch_processed = df['text_processed'].iloc[start_idx:end_idx]
        batch_tfidf = tfidf_vectorizer.transform(batch_processed)
        tfidf_vectors.extend(batch_tfidf.toarray())
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            print(f"    Batch {batch_idx + 1}/{num_batches} processed", end='\r')
    
    df['tfidf'] = tfidf_vectors
    print()  # New line
    
    # Drop processed text column
    df = df.drop(columns=['text_processed'])
    
    # Save to output directory
    base_name = file_name.replace('_news.parquet', '')
    output_name = f"{base_name}_traditional.parquet"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    
    # Save vectorizers for future use
    bow_vectorizer_path = os.path.join(OUTPUT_DIR, f"{base_name}_bow_vectorizer.pkl")
    tfidf_vectorizer_path = os.path.join(OUTPUT_DIR, f"{base_name}_tfidf_vectorizer.pkl")
    
    with open(bow_vectorizer_path, 'wb') as f:
        pickle.dump(bow_vectorizer, f)
    with open(tfidf_vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    bow_dim = len(bow_vectors[0])
    tfidf_dim = len(tfidf_vectors[0])
    print(f"  ✓ Saved {output_name} with {len(df)} records")
    print(f"    BoW dimension: {bow_dim}, TF-IDF dimension: {tfidf_dim}\n")

print("All files processed successfully!")

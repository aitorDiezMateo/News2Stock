import pandas as pd
import os
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Descarga el tokenizer de NLTK
nltk.download('punkt')

# Rutas de archivos
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'summarized')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Ruta al modelo Glove
glove_model_path = os.path.join(MODELS_DIR, 'glove.2024.wikigiga.300d.txt')

# Dataset
df = pd.read_parquet(os.path.join(SUMMARIES_DIR, 'nvidia_news.parquet'), engine="pyarrow")
df["text"] = df["title"] + "\n\n" + df["body_summary"]
df = df.drop(columns=["clean_body", "body_summary", "teaser","title","author"])

# Load GloVe model
tmp_file = os.path.join(MODELS_DIR, 'glove.2024.wikigiga.300d.word2vec.txt')
print("Converting GloVe to Word2Vec format...")
glove2word2vec(glove_model_path, tmp_file)  # Converts GloVe -> Word2Vec format
print("Loading GloVe model...")
glove_model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
print(f"Model loaded! Vocabulary size: {len(glove_model)}")

# Tokenize all text and collect unique words
print("\nTokenizing corpus...")
all_words = []
for text in df['text']:
    tokens = word_tokenize(text.lower())  # Convert to lowercase for matching
    all_words.extend(tokens)

# Get unique words and their counts
unique_words = set(all_words)
total_words = len(all_words)
unique_word_count = len(unique_words)

print(f"Total words in corpus: {total_words:,}")
print(f"Unique words in corpus: {unique_word_count:,}")

# Check which words are in GloVe vocabulary
words_in_glove = set()
words_not_in_glove = set()

for word in unique_words:
    if word in glove_model:
        words_in_glove.add(word)
    else:
        words_not_in_glove.add(word)

# Calculate percentages
percentage_not_in_glove = (len(words_not_in_glove) / unique_word_count) * 100
percentage_in_glove = (len(words_in_glove) / unique_word_count) * 100

# Calculate coverage by token count (weighted by frequency)
tokens_not_in_glove = sum(1 for word in all_words if word not in glove_model)
token_coverage = ((total_words - tokens_not_in_glove) / total_words) * 100

print("\n" + "="*60)
print("GLOVE VOCABULARY COVERAGE")
print("="*60)
print(f"\nUnique words in GloVe: {len(words_in_glove):,} ({percentage_in_glove:.2f}%)")
print(f"Unique words NOT in GloVe: {len(words_not_in_glove):,} ({percentage_not_in_glove:.2f}%)")
print(f"\nToken coverage (by frequency): {token_coverage:.2f}%")
print(f"Tokens not covered: {tokens_not_in_glove:,} out of {total_words:,}")

# Show some examples of words not in GloVe
print("\n" + "="*60)
print("SAMPLE OF WORDS NOT IN GLOVE (first 20):")
print("="*60)
for i, word in enumerate(list(words_not_in_glove)[:20]):
    print(f"{i+1}. '{word}'")


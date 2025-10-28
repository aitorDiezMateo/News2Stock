import os
from collections import Counter
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk

nltk.download('punkt')

# Paths
ROOT_DIR = os.getcwd()
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'summarized')
GLOVE_PATH = os.path.join(ROOT_DIR, 'models', 'glove.2024.wikigiga.300d.txt')

# Load dataset
df = pd.read_parquet(os.path.join(SUMMARIES_DIR, 'nvidia_news.parquet'), engine="pyarrow")
df["text"] = df["title"] + "\n\n" + df["body_summary"]
df = df.drop(columns=["clean_body", "body_summary", "teaser","title","author"])

# Tokenize all text
all_words = []
for text in df['text']:
    tokens = word_tokenize(text.lower())
    all_words.extend(tokens)

word_counts = Counter(all_words)
unique_words = set(word_counts.keys())

# Load GloVe vocabulary only (ignore vectors)
glove_vocab = set()
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.split(' ')[0]  # First element is the word
        glove_vocab.add(word)

# Compare
words_in_glove = unique_words & glove_vocab
words_not_in_glove = unique_words - glove_vocab

total_words = len(all_words)
tokens_not_in_glove = sum(count for word, count in word_counts.items() if word not in glove_vocab)
token_coverage = ((total_words - tokens_not_in_glove) / total_words) * 100

print(f"Total words in corpus: {total_words:,}")
print(f"Unique words in corpus: {len(unique_words):,}")
print(f"Unique words in GloVe: {len(words_in_glove):,}")
print(f"Unique words NOT in GloVe: {len(words_not_in_glove):,}")
print(f"Token coverage (by frequency): {token_coverage:.2f}%")

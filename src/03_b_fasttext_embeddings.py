import fasttext.util
import pandas as pd
import os
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

DOWNLOAD = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'summarized')
FASTTEXT_PATH = os.path.join(ROOT_DIR, 'models', 'cc.en.300.bin')

if DOWNLOAD:
    fasttext.util.download_model('en', if_exists='ignore')  # English

ft = fasttext.load_model(FASTTEXT_PATH)
print("FastText model loaded.")

# Load dataset
df = pd.read_parquet(os.path.join(SUMMARIES_DIR, 'nvidia_news.parquet'), engine="pyarrow")
df["text"] = df["title"] + "\n\n" + df["body_summary"]
df = df.drop(columns=["clean_body", "body_summary", "teaser", "title", "author"])

# Tokenize all text
all_words = []
for text in df['text']:
    tokens = word_tokenize(text.lower())
    all_words.extend(tokens)

word_counts = Counter(all_words)
unique_words = set(word_counts.keys())

# Get FastText vocabulary
fasttext_vocab = set(ft.get_words())

# Compare
words_in_fasttext = unique_words & fasttext_vocab
words_not_in_fasttext = unique_words - fasttext_vocab

total_words = len(all_words)
tokens_not_in_fasttext = sum(count for word, count in word_counts.items() if word not in fasttext_vocab)
token_coverage = ((total_words - tokens_not_in_fasttext) / total_words) * 100

print(f"Total words in corpus: {total_words:,}")
print(f"Unique words in corpus: {len(unique_words):,}")
print(f"Unique words in FastText: {len(words_in_fasttext):,}")
print(f"Unique words NOT in FastText: {len(words_not_in_fasttext):,}")
print(f"Token coverage (by frequency): {token_coverage:.2f}%") 

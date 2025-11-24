"""
Configuration file for simple seq2seq text summarization model
"""

import os

# ============================================================================
# SPECIAL TOKENS
# ============================================================================
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'  # Start of sequence
EOS_TOKEN = '<EOS>'  # End of sequence

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
EMBEDDING_DIM = 128      # Reduced from 256 to save memory
HIDDEN_DIM = 256         # Reduced from 512 to save memory
NUM_LAYERS = 1           # Reduced from 2 to save memory
DROPOUT = 0.3
BIDIRECTIONAL = True     # For encoder only

# ============================================================================
# DATA PARAMETERS
# ============================================================================
MAX_SOURCE_LEN = 300     # Reduced from 400 to save memory
MAX_TARGET_LEN = 80      # Reduced from 100 to save memory
MIN_WORD_FREQ = 2        # Minimum frequency for word to be in vocabulary
VOCAB_SIZE = 20000       # Reduced from 30000 to save memory

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
BATCH_SIZE = 8           # Reduced from 16 to save memory
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5  # Probability of using teacher forcing
CLIP_GRAD = 5.0              # Gradient clipping threshold

# ============================================================================
# PATHS
# ============================================================================
# Go up two levels: simple_seq2seq -> summaries -> News2Stock
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_DIR, 'data/news/summarized/apple_news.parquet')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

# ============================================================================
# DEVICE
# ============================================================================
USE_CUDA = True  # Will check availability at runtime


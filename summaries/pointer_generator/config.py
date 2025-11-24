"""
Configuration file for Pointer-Generator Network text summarization model
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
EMBEDDING_DIM = 128      # Dimension of word embeddings
HIDDEN_DIM = 256         # LSTM hidden dimension
NUM_LAYERS = 1           # Number of LSTM layers
DROPOUT = 0.3
BIDIRECTIONAL = True     # For encoder only

# Pointer-Generator specific parameters
USE_COVERAGE = False      # Enable coverage mechanism to prevent repetition
COVERAGE_LAMBDA = 0.1    # Weight for coverage loss (reduced from 1.0 to prevent overfitting)

# ============================================================================
# DATA PARAMETERS
# ============================================================================
MAX_SOURCE_LEN = 300     # Maximum source sequence length
MAX_TARGET_LEN = 80      # Maximum target sequence length
MIN_WORD_FREQ = 2        # Minimum frequency for word to be in vocabulary
VOCAB_SIZE = 20000       # Maximum vocabulary size

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
BATCH_SIZE = 16           # Batch size for training
NUM_EPOCHS = 20          # Number of training epochs
LEARNING_RATE = 0.001    # Learning rate
TEACHER_FORCING_RATIO = 0.5  # Probability of using teacher forcing
CLIP_GRAD = 5.0          # Gradient clipping threshold

# Learning rate scheduling
USE_LR_SCHEDULER = True
LR_DECAY_FACTOR = 0.5    # Factor to reduce learning rate
LR_PATIENCE = 2          # Epochs with no improvement after which LR will be reduced

# ============================================================================
# PATHS
# ============================================================================
# Go up two levels: pointer_generator -> summaries -> News2Stock
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_DIR, 'data/news/summarized/apple_news.parquet')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

# ============================================================================
# DEVICE
# ============================================================================
USE_CUDA = True  # Will check availability at runtime

# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================
BEAM_SIZE = 4            # Beam size for beam search during inference
MIN_DECODE_STEPS = 10    # Minimum number of decoding steps
MAX_DECODE_STEPS = 80    # Maximum number of decoding steps


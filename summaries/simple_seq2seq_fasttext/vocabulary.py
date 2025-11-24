"""
Vocabulary class for managing word-to-index mappings
"""

from collections import Counter
from typing import List
import config


class Vocabulary:
    """Simple vocabulary class for text processing"""
    
    def __init__(self):
        self.word2idx = {
            config.PAD_TOKEN: config.PAD_IDX,
            config.UNK_TOKEN: config.UNK_IDX,
            config.SOS_TOKEN: config.SOS_IDX,
            config.EOS_TOKEN: config.EOS_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_counts = Counter()
        
    def build_vocabulary(self, texts: List[List[str]], min_freq: int = 2, max_size: int = None):
        """
        Build vocabulary from list of tokenized texts
        
        Args:
            texts: List of tokenized texts (each text is a list of words)
            min_freq: Minimum frequency for a word to be included
            max_size: Maximum vocabulary size
        """
        # Count word frequencies
        for text in texts:
            self.word_counts.update(text)
        
        # Sort by frequency and add to vocabulary
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words:
            if count < min_freq:
                break
            if max_size and len(self.word2idx) >= max_size:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Vocabulary built with {len(self.word2idx)} words")
        print(f"Most common words: {sorted_words[:10]}")
    
    def encode(self, text: List[str]) -> List[int]:
        """Convert list of words to list of indices"""
        return [self.word2idx.get(word, config.UNK_IDX) for word in text]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert list of indices to list of words"""
        return [self.idx2word.get(idx, config.UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return len(self.word2idx)


"""
Vocabulary class for Pointer-Generator Network with OOV handling
"""

from collections import Counter
from typing import List, Tuple
import config


class Vocabulary:
    """Vocabulary class with support for OOV words (for pointer mechanism)"""
    
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
    
    def encode_with_oov(self, text: List[str]) -> Tuple[List[int], List[str]]:
        """
        Encode text with OOV handling for pointer mechanism
        
        Args:
            text: List of words
            
        Returns:
            indices: List of indices (OOV words get temporary indices >= vocab_size)
            oov_words: List of OOV words in order of appearance
        """
        indices = []
        oov_words = []
        
        for word in text:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                # Assign temporary OOV index
                if word not in oov_words:
                    oov_words.append(word)
                oov_idx = len(self.word2idx) + oov_words.index(word)
                indices.append(oov_idx)
        
        return indices, oov_words
    
    def decode_with_oov(self, indices: List[int], oov_words: List[str]) -> List[str]:
        """
        Decode indices with OOV handling
        
        Args:
            indices: List of indices
            oov_words: List of OOV words from source
            
        Returns:
            words: List of decoded words
        """
        words = []
        vocab_size = len(self.word2idx)
        
        for idx in indices:
            if idx < vocab_size:
                words.append(self.idx2word.get(idx, config.UNK_TOKEN))
            else:
                # OOV word from source
                oov_idx = idx - vocab_size
                if oov_idx < len(oov_words):
                    words.append(oov_words[oov_idx])
                else:
                    words.append(config.UNK_TOKEN)
        
        return words
    
    def __len__(self):
        return len(self.word2idx)


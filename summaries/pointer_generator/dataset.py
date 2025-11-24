"""
Dataset and data loading utilities for Pointer-Generator Network
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import config


class PointerGeneratorDataset(Dataset):
    """Dataset for Pointer-Generator Network with OOV handling"""
    
    def __init__(self, sources, targets, vocab):
        """
        Args:
            sources: List of tokenized source texts
            targets: List of tokenized target summaries
            vocab: Vocabulary object
        """
        self.sources = sources
        self.targets = targets
        self.vocab = vocab
        
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        """
        Returns:
            src: List of token indices (in vocab)
            src_extended: List of token indices (with OOV)
            trg: List of token indices (in vocab)
            trg_extended: List of token indices (with OOV from source)
            oov_words: List of OOV words from source
        """
        src_tokens = self.sources[idx]
        trg_tokens = self.targets[idx]
        
        # Encode source with OOV handling
        src_extended, oov_words = self.vocab.encode_with_oov(src_tokens)
        
        # Encode source without OOV (for embedding lookup)
        src = self.vocab.encode(src_tokens)
        
        # Encode target with OOV handling (using source OOV words)
        trg_extended = []
        for word in trg_tokens:
            if word in self.vocab.word2idx:
                trg_extended.append(self.vocab.word2idx[word])
            elif word in oov_words:
                # Use extended vocab index
                oov_idx = len(self.vocab) + oov_words.index(word)
                trg_extended.append(oov_idx)
            else:
                # Word not in vocab and not in source - use UNK
                trg_extended.append(config.UNK_IDX)
        
        # Encode target without OOV (for teacher forcing input)
        trg = self.vocab.encode(trg_tokens)
        
        # Add SOS and EOS to target
        trg = [config.SOS_IDX] + trg + [config.EOS_IDX]
        trg_extended = [config.SOS_IDX] + trg_extended + [config.EOS_IDX]
        
        # Truncate if too long
        src = src[:config.MAX_SOURCE_LEN]
        src_extended = src_extended[:config.MAX_SOURCE_LEN]
        trg = trg[:config.MAX_TARGET_LEN]
        trg_extended = trg_extended[:config.MAX_TARGET_LEN]
        
        return (
            torch.LongTensor(src),
            torch.LongTensor(src_extended),
            torch.LongTensor(trg),
            torch.LongTensor(trg_extended),
            oov_words
        )


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch
    
    Args:
        batch: List of (src, src_extended, trg, trg_extended, oov_words) tuples
    Returns:
        src_padded: [batch_size, max_src_len]
        src_extended_padded: [batch_size, max_src_len]
        src_lengths: [batch_size]
        trg_padded: [batch_size, max_trg_len]
        trg_extended_padded: [batch_size, max_trg_len]
        oov_size: int (maximum number of OOV words in batch)
        oov_lists: List of OOV word lists for each example
    """
    # Separate components
    sources, sources_extended, targets, targets_extended, oov_lists = zip(*batch)
    
    # Get lengths before padding
    src_lengths = torch.LongTensor([len(src) for src in sources])
    
    # Pad sequences
    src_padded = pad_sequence(sources, batch_first=True, padding_value=config.PAD_IDX)
    src_extended_padded = pad_sequence(sources_extended, batch_first=True, padding_value=config.PAD_IDX)
    trg_padded = pad_sequence(targets, batch_first=True, padding_value=config.PAD_IDX)
    trg_extended_padded = pad_sequence(targets_extended, batch_first=True, padding_value=config.PAD_IDX)
    
    # Get maximum OOV size in batch
    oov_size = max([len(oov) for oov in oov_lists]) if oov_lists else 0
    
    return (
        src_padded,
        src_extended_padded,
        src_lengths,
        trg_padded,
        trg_extended_padded,
        oov_size,
        list(oov_lists)
    )


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """Create DataLoader with custom collate function"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


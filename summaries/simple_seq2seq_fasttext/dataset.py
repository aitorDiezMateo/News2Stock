"""
Dataset and data loading utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import config


class SummarizationDataset(Dataset):
    """Dataset for text summarization"""
    
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
            src: List of token indices
            trg: List of token indices (with SOS and EOS)
        """
        src = self.sources[idx]
        trg = self.targets[idx]
        
        # Encode to indices
        src_indices = self.vocab.encode(src)
        trg_indices = self.vocab.encode(trg)
        
        # Add SOS and EOS to target
        trg_indices = [config.SOS_IDX] + trg_indices + [config.EOS_IDX]
        
        # Truncate if too long
        src_indices = src_indices[:config.MAX_SOURCE_LEN]
        trg_indices = trg_indices[:config.MAX_TARGET_LEN]
        
        return torch.LongTensor(src_indices), torch.LongTensor(trg_indices)


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch
    
    Args:
        batch: List of (src, trg) tuples
    Returns:
        src_padded: [batch_size, max_src_len]
        src_lengths: [batch_size]
        trg_padded: [batch_size, max_trg_len]
    """
    # Separate sources and targets
    sources, targets = zip(*batch)
    
    # Get lengths before padding
    src_lengths = torch.LongTensor([len(src) for src in sources])
    
    # Pad sequences
    src_padded = pad_sequence(sources, batch_first=True, padding_value=config.PAD_IDX)
    trg_padded = pad_sequence(targets, batch_first=True, padding_value=config.PAD_IDX)
    
    return src_padded, src_lengths, trg_padded


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """Create DataLoader with custom collate function"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


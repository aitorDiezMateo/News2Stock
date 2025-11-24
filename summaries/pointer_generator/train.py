"""
Training script for Pointer-Generator Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
import os
from datetime import datetime

import config
from model import Encoder, Decoder, Attention, PointerGeneratorNetwork, calculate_loss
from dataset import PointerGeneratorDataset, get_dataloader


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, clip, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_nll_loss = 0
    epoch_coverage_loss = 0
    
    for i, (src, src_extended, src_lengths, trg, trg_extended, oov_size, oov_lists) in enumerate(dataloader):
        src = src.to(device)
        src_extended = src_extended.to(device)
        src_lengths = src_lengths.to(device)
        trg = trg.to(device)
        trg_extended = trg_extended.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        final_dists, coverages = model(
            src, src_lengths, trg, src_extended, oov_size, 
            teacher_forcing_ratio=config.TEACHER_FORCING_RATIO
        )
        
        # Create padding mask for target
        padding_mask = (trg != config.PAD_IDX).float()
        
        # Extract attention weights from coverages for loss calculation
        # coverages contains cumulative attention, we need individual attention weights
        attention_weights_list = []
        if len(coverages) > 0:
            prev_coverage = None
            for coverage in coverages:
                if prev_coverage is None:
                    attention_weights_list.append(coverage)
                else:
                    attention_weights_list.append(coverage - prev_coverage)
                prev_coverage = coverage
        
        # Calculate loss
        loss, nll_loss, coverage_loss = calculate_loss(
            final_dists, trg_extended, coverages, attention_weights_list, padding_mask
        )
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_nll_loss += nll_loss.item()
        if isinstance(coverage_loss, torch.Tensor):
            epoch_coverage_loss += coverage_loss.item()
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f'  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}, '
                  f'NLL: {nll_loss.item():.4f}, Coverage: {coverage_loss if isinstance(coverage_loss, torch.Tensor) else 0:.4f}')
    
    return (
        epoch_loss / len(dataloader),
        epoch_nll_loss / len(dataloader),
        epoch_coverage_loss / len(dataloader)
    )


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set"""
    model.eval()
    epoch_loss = 0
    epoch_nll_loss = 0
    epoch_coverage_loss = 0
    
    with torch.no_grad():
        for src, src_extended, src_lengths, trg, trg_extended, oov_size, oov_lists in dataloader:
            src = src.to(device)
            src_extended = src_extended.to(device)
            src_lengths = src_lengths.to(device)
            trg = trg.to(device)
            trg_extended = trg_extended.to(device)
            
            # Forward pass without teacher forcing
            final_dists, coverages = model(
                src, src_lengths, trg, src_extended, oov_size, 
                teacher_forcing_ratio=0
            )
            
            # Create padding mask for target
            padding_mask = (trg != config.PAD_IDX).float()
            
            # Extract attention weights from coverages
            attention_weights_list = []
            if len(coverages) > 0:
                prev_coverage = None
                for coverage in coverages:
                    if prev_coverage is None:
                        attention_weights_list.append(coverage)
                    else:
                        attention_weights_list.append(coverage - prev_coverage)
                    prev_coverage = coverage
            
            # Calculate loss
            loss, nll_loss, coverage_loss = calculate_loss(
                final_dists, trg_extended, coverages, attention_weights_list, padding_mask
            )
            
            epoch_loss += loss.item()
            epoch_nll_loss += nll_loss.item()
            if isinstance(coverage_loss, torch.Tensor):
                epoch_coverage_loss += coverage_loss.item()
    
    return (
        epoch_loss / len(dataloader),
        epoch_nll_loss / len(dataloader),
        epoch_coverage_loss / len(dataloader)
    )


def epoch_time(start_time, end_time):
    """Calculate elapsed time"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, vocab_size, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab_size': vocab_size
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f'  ✓ Saved checkpoint to {checkpoint_path}')
    return checkpoint_path


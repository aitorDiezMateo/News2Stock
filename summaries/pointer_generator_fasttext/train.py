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
from model import Encoder, Decoder, Attention, PointerGeneratorNetwork, calculate_loss, init_embeddings_with_fasttext
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("POINTER-GENERATOR NETWORK TRAINING (FastText)")
print("="*80)

# Set device
device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")

# Load preprocessed data
print("\nLoading preprocessed data...")
data_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data.pkl')
with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

train_sources, train_targets = data_dict['train']
val_sources, val_targets = data_dict['val']
vocab = data_dict['vocab']

print(f"✓ Train: {len(train_sources)} examples")
print(f"✓ Val: {len(val_sources)} examples")
print(f"✓ Vocabulary size: {len(vocab)}")

# Create datasets
train_dataset = PointerGeneratorDataset(train_sources, train_targets, vocab)
val_dataset = PointerGeneratorDataset(val_sources, val_targets, vocab)

# Create dataloaders
train_loader = get_dataloader(train_dataset, config.BATCH_SIZE, shuffle=True)
val_loader = get_dataloader(val_dataset, config.BATCH_SIZE, shuffle=False)

print(f"\n✓ Created dataloaders (batch_size={config.BATCH_SIZE})")

# Initialize model
print("\n" + "="*80)
print("INITIALIZING MODEL")
print("="*80)

# Encoder (always bidirectional)
encoder = Encoder(
    vocab_size=len(vocab),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT
)

# Attention
encoder_hidden_dim = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
attention = Attention(encoder_hidden_dim, config.HIDDEN_DIM)

# Decoder
decoder = Decoder(
    vocab_size=len(vocab),
    embedding_dim=config.EMBEDDING_DIM,
    encoder_hidden_dim=encoder_hidden_dim,
    decoder_hidden_dim=config.HIDDEN_DIM,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    attention=attention
)

# Complete model
model = PointerGeneratorNetwork(encoder, decoder, device).to(device)

# Initialize embeddings with FastText if enabled
if config.USE_FASTTEXT:
    print("\nInitializing embeddings with FastText...")
    init_embeddings_with_fasttext(encoder.embedding, vocab, config.FASTTEXT_MODEL_PATH, config.EMBEDDING_DIM)
    init_embeddings_with_fasttext(decoder.embedding, vocab, config.FASTTEXT_MODEL_PATH, config.EMBEDDING_DIM)
    print("✓ FastText embeddings loaded")

print(f"✓ Model initialized")
print(f"✓ Trainable parameters: {count_parameters(model):,}")
print(f"✓ Coverage mechanism: {'Enabled' if config.USE_COVERAGE else 'Disabled'}")

# Initialize optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.LR_DECAY_FACTOR, patience=config.LR_PATIENCE)

print(f"✓ Optimizer: Adam (lr={config.LEARNING_RATE})")
print(f"✓ LR Scheduler: ReduceLROnPlateau (patience={config.LR_PATIENCE})")

# Training loop
print("\n" + "="*80)
print("TRAINING")
print("="*80)

best_val_loss = float('inf')
best_model_path = None

# Create log file
log_dir = config.LOG_DIR
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

with open(log_file, 'w') as f:
    f.write("Epoch,Train_Loss,Train_NLL,Train_Cov,Val_Loss,Val_NLL,Val_Cov,Time(s)\n")

print(f"✓ Logging to: {log_file}\n")

for epoch in range(config.NUM_EPOCHS):
    start_time = time.time()
    
    print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    print("-" * 40)
    
    # Train
    train_loss, train_nll, train_cov = train_epoch(model, train_loader, optimizer, config.CLIP_GRAD, device)
    
    # Validate
    val_loss, val_nll, val_cov = evaluate(model, val_loader, device)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # Update learning rate
    if config.USE_LR_SCHEDULER:
        scheduler.step(val_loss)
    
    # Log results
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1},{train_loss:.4f},{train_nll:.4f},{train_cov:.4f},{val_loss:.4f},{val_nll:.4f},{val_cov:.4f},{end_time-start_time:.2f}\n")
    
    print(f"\n  Train Loss: {train_loss:.4f} (NLL: {train_nll:.4f}, Coverage: {train_cov:.4f})")
    print(f"  Val Loss: {val_loss:.4f} (NLL: {val_nll:.4f}, Coverage: {val_cov:.4f})")
    print(f"  Time: {epoch_mins}m {epoch_secs}s")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss, 
                       len(vocab), config.CHECKPOINT_DIR)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_size': len(vocab)
        }, best_model_path)
        print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
    
    print()

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved to: {best_model_path}")
print(f"Training log saved to: {log_file}")
print("\nNote: Apple news is reserved for inference/testing in a separate script.")

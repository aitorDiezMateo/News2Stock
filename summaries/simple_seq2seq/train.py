"""
Training script for simple seq2seq summarization model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
import os
from datetime import datetime

import config
from model import Encoder, Decoder, Attention, Seq2Seq
from dataset import SummarizationDataset, get_dataloader


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for i, (src, src_lengths, trg) in enumerate(dataloader):
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_lengths, trg, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO)
        
        # output: [batch_size, trg_len, vocab_size]
        # trg: [batch_size, trg_len]
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)  # Exclude first token (SOS)
        trg = trg[:, 1:].contiguous().view(-1)  # Exclude first token (SOS)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f'  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, src_lengths, trg in dataloader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            trg = trg.to(device)
            
            # Forward pass without teacher forcing
            output = model(src, src_lengths, trg, teacher_forcing_ratio=0)
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


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


print("="*80)
print("SIMPLE SEQ2SEQ TRAINING")
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
test_sources, test_targets = data_dict['test']
vocab = data_dict['vocab']

print(f"✓ Train: {len(train_sources)} examples")
print(f"✓ Val: {len(val_sources)} examples")
print(f"✓ Test: {len(test_sources)} examples")
print(f"✓ Vocabulary size: {len(vocab)}")

# Create datasets
train_dataset = SummarizationDataset(train_sources, train_targets, vocab)
val_dataset = SummarizationDataset(val_sources, val_targets, vocab)
test_dataset = SummarizationDataset(test_sources, test_targets, vocab)

# Create dataloaders
train_loader = get_dataloader(train_dataset, config.BATCH_SIZE, shuffle=True)
val_loader = get_dataloader(val_dataset, config.BATCH_SIZE, shuffle=False)
test_loader = get_dataloader(test_dataset, config.BATCH_SIZE, shuffle=False)

print(f"\n✓ Created dataloaders (batch_size={config.BATCH_SIZE})")
    
# Initialize model
print("\n" + "="*80)
print("INITIALIZING MODEL")
print("="*80)
    
# Encoder
encoder = Encoder(
    vocab_size=len(vocab),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    bidirectional=config.BIDIRECTIONAL
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
model = Seq2Seq(encoder, decoder, device).to(device)
    
print(f"✓ Model initialized")
print(f"✓ Trainable parameters: {count_parameters(model):,}")
    
# Initialize optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    
print(f"✓ Optimizer: Adam (lr={config.LEARNING_RATE})")
print(f"✓ Loss: CrossEntropyLoss")
    
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
    f.write("Epoch,Train_Loss,Val_Loss,Time(s)\n")

print(f"✓ Logging to: {log_file}\n")

for epoch in range(config.NUM_EPOCHS):
    start_time = time.time()

    print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    print("-" * 40)

# Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, config.CLIP_GRAD, device)

# Validate
    val_loss = evaluate(model, val_loader, criterion, device)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

# Log results
    with open(log_file, 'a') as f:
    f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{end_time-start_time:.2f}\n")

    print(f"\n  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Time: {epoch_mins}m {epoch_secs}s")

# Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
    save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss, 
                  len(vocab), config.CHECKPOINT_DIR)

# Save best model
    if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_path = save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss,
                                     len(vocab), config.CHECKPOINT_DIR)
    best_model_path = best_model_path.replace(f'checkpoint_epoch_{epoch+1}.pt', 'best_model.pt')
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

# Final evaluation on test set
    print("="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

# Load best model
    if best_model_path and os.path.exists(best_model_path):
    print(f"Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
print(f"Training log saved to: {log_file}")


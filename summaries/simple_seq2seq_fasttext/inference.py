"""
Inference script for generating summaries with the trained model
"""

import torch
import pickle
import argparse
import config
from model import Encoder, Decoder, Attention, Seq2Seq
from preprocess import clean_text, tokenize


def load_model(checkpoint_path, vocab, device):
    """Load trained model from checkpoint"""
    # Initialize model architecture
    encoder = Encoder(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL
    )
    
    encoder_hidden_dim = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
    attention = Attention(encoder_hidden_dim, config.HIDDEN_DIM)
    
    decoder = Decoder(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        attention=attention
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Load weights (checkpoint already contains trained embeddings)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model


def generate_summary(model, source_text, vocab, device, max_length=100):
    """
    Generate summary for a given source text
    
    Args:
        model: Trained Seq2Seq model
        source_text: Input text string
        vocab: Vocabulary object
        device: torch device
        max_length: Maximum length of generated summary
    
    Returns:
        summary: Generated summary as string
    """
    model.eval()
    
    # Preprocess source text
    cleaned = clean_text(source_text)
    tokens = tokenize(cleaned)
    
    # Encode to indices
    src_indices = vocab.encode(tokens)
    src_indices = src_indices[:config.MAX_SOURCE_LEN]  # Truncate if too long
    
    # Convert to tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    src_lengths = torch.LongTensor([len(src_indices)]).to(device)
    
    with torch.no_grad():
        # Encode
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_lengths)
        
        # Project encoder hidden state if needed
        if model.bridge_h is not None:
            hidden = model.bridge_h(hidden)
            cell = model.bridge_c(cell)
        
        # Create mask
        mask = (src_tensor != config.PAD_IDX)
        
        # Start with SOS token
        trg_indices = [config.SOS_IDX]
        
        # Generate tokens one by one
        for _ in range(max_length):
            trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
            
            # Decode one step
            output, hidden, cell, attention_weights = model.decoder(
                trg_tensor, hidden, cell, encoder_outputs, mask
            )
            
            # Get predicted token
            pred_token = output.argmax(1).item()
            
            # Add to sequence
            trg_indices.append(pred_token)
            
            # Stop if EOS token is generated
            if pred_token == config.EOS_IDX:
                break
        
        # Decode indices to words
        summary_tokens = vocab.decode(trg_indices[1:-1])  # Exclude SOS and EOS
        summary = ' '.join(summary_tokens)
    
    return summary


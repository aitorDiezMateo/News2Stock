"""
Inference script for Pointer-Generator Network with beam search
"""

import torch
import pickle
import argparse
import config
from model import Encoder, Decoder, Attention, PointerGeneratorNetwork
from preprocess import clean_text, tokenize
import numpy as np


class Beam:
    """Beam search helper class"""
    
    def __init__(self, tokens, log_probs, hidden, cell, coverage):
        self.tokens = tokens  # List of token indices
        self.log_probs = log_probs  # List of log probabilities
        self.hidden = hidden  # Decoder hidden state
        self.cell = cell  # Decoder cell state
        self.coverage = coverage  # Coverage vector
        
    @property
    def avg_log_prob(self):
        """Average log probability"""
        return sum(self.log_probs) / len(self.tokens)
    
    @property
    def latest_token(self):
        """Most recent token"""
        return self.tokens[-1]
    
    def extend(self, token, log_prob, hidden, cell, coverage):
        """Extend beam with new token"""
        return Beam(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            hidden=hidden,
            cell=cell,
            coverage=coverage
        )


def load_model(checkpoint_path, vocab, device):
    """Load trained model from checkpoint"""
    # Initialize model architecture
    encoder = Encoder(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    encoder_hidden_dim = config.HIDDEN_DIM * 2  # Bidirectional
    attention = Attention(encoder_hidden_dim, config.HIDDEN_DIM, use_coverage=config.USE_COVERAGE)
    
    decoder = Decoder(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        attention=attention
    )
    
    model = PointerGeneratorNetwork(encoder, decoder, device).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model


def beam_search(model, source_text, vocab, device, beam_size=4):
    """
    Generate summary using beam search
    
    Args:
        model: Trained PointerGeneratorNetwork
        source_text: Input text string
        vocab: Vocabulary object
        device: torch device
        beam_size: Beam size for beam search
    
    Returns:
        summary: Generated summary as string
    """
    model.eval()
    
    # Preprocess source text
    cleaned = clean_text(source_text)
    tokens = tokenize(cleaned)
    
    # Encode with OOV handling
    src_extended, oov_words = vocab.encode_with_oov(tokens)
    src = vocab.encode(tokens)
    
    # Truncate if too long
    src = src[:config.MAX_SOURCE_LEN]
    src_extended = src_extended[:config.MAX_SOURCE_LEN]
    
    # Convert to tensors
    src_tensor = torch.LongTensor(src).unsqueeze(0).to(device)  # [1, src_len]
    src_extended_tensor = torch.LongTensor(src_extended).unsqueeze(0).to(device)
    src_lengths = torch.LongTensor([len(src)]).to(device)
    
    with torch.no_grad():
        # Encode
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_lengths)
        
        # Create mask
        mask = (src_tensor != config.PAD_IDX)
        
        # Initialize beams
        beams = [Beam(
            tokens=[config.SOS_IDX],
            log_probs=[0.0],
            hidden=hidden,
            cell=cell,
            coverage=None
        )]
        
        # Beam search
        for step in range(config.MAX_DECODE_STEPS):
            # Generate candidates from all beams
            all_candidates = []
            
            for beam in beams:
                # Skip if beam has ended
                if beam.latest_token == config.EOS_IDX:
                    all_candidates.append(beam)
                    continue
                
                # Prepare input
                input_token = beam.latest_token
                # If token is from extended vocab, use UNK for decoder input
                if input_token >= len(vocab):
                    input_token = config.UNK_IDX
                
                input_tensor = torch.LongTensor([input_token]).to(device)
                
                # Decode one step
                p_vocab, p_gen, new_hidden, new_cell, attention_weights, new_coverage = model.decoder(
                    input_tensor, beam.hidden, beam.cell, encoder_outputs, mask, beam.coverage
                )
                
                # Calculate final distribution
                vocab_size = len(vocab)
                oov_size = len(oov_words)
                extended_vocab_size = vocab_size + oov_size
                
                # Weighted vocabulary distribution
                p_vocab_weighted = p_gen * p_vocab  # [1, vocab_size]
                
                # Weighted attention distribution
                p_copy_weighted = (1 - p_gen) * attention_weights  # [1, src_len]
                
                # Create extended vocabulary distribution
                final_dist = torch.zeros(1, extended_vocab_size).to(device)
                final_dist[:, :vocab_size] = p_vocab_weighted
                final_dist.scatter_add_(1, src_extended_tensor, p_copy_weighted)
                
                # Get top k tokens
                log_probs = torch.log(final_dist + 1e-10).squeeze(0)  # [extended_vocab_size]
                top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_size)
                
                # Create new beams
                for i in range(beam_size):
                    token = top_k_indices[i].item()
                    log_prob = top_k_log_probs[i].item()
                    
                    new_beam = beam.extend(token, log_prob, new_hidden, new_cell, new_coverage)
                    all_candidates.append(new_beam)
            
            # Sort all candidates by average log probability
            all_candidates.sort(key=lambda x: x.avg_log_prob, reverse=True)
            
            # Select top beams
            beams = all_candidates[:beam_size]
            
            # Check if all beams have ended
            if all(beam.latest_token == config.EOS_IDX for beam in beams):
                break
            
            # Stop if minimum decode steps reached and best beam has ended
            if step >= config.MIN_DECODE_STEPS and beams[0].latest_token == config.EOS_IDX:
                break
        
        # Select best beam
        best_beam = beams[0]
        
        # Decode tokens (exclude SOS and EOS)
        output_tokens = best_beam.tokens[1:]  # Remove SOS
        if output_tokens and output_tokens[-1] == config.EOS_IDX:
            output_tokens = output_tokens[:-1]  # Remove EOS
        
        # Decode with OOV handling
        summary_words = vocab.decode_with_oov(output_tokens, oov_words)
        summary = ' '.join(summary_words)
    
    return summary


def generate_summary_greedy(model, source_text, vocab, device):
    """
    Generate summary using greedy decoding (faster but lower quality)
    
    Args:
        model: Trained PointerGeneratorNetwork
        source_text: Input text string
        vocab: Vocabulary object
        device: torch device
    
    Returns:
        summary: Generated summary as string
    """
    model.eval()
    
    # Preprocess source text
    cleaned = clean_text(source_text)
    tokens = tokenize(cleaned)
    
    # Encode with OOV handling
    src_extended, oov_words = vocab.encode_with_oov(tokens)
    src = vocab.encode(tokens)
    
    # Truncate if too long
    src = src[:config.MAX_SOURCE_LEN]
    src_extended = src_extended[:config.MAX_SOURCE_LEN]
    
    # Convert to tensors
    src_tensor = torch.LongTensor(src).unsqueeze(0).to(device)  # [1, src_len]
    src_extended_tensor = torch.LongTensor(src_extended).unsqueeze(0).to(device)
    src_lengths = torch.LongTensor([len(src)]).to(device)
    
    with torch.no_grad():
        # Encode
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_lengths)
        
        # Create mask
        mask = (src_tensor != config.PAD_IDX)
        
        # Initialize
        coverage = None
        output_tokens = []
        input_token = config.SOS_IDX
        
        # Generate tokens one by one
        for _ in range(config.MAX_DECODE_STEPS):
            # Prepare input
            if input_token >= len(vocab):
                input_token = config.UNK_IDX
            
            input_tensor = torch.LongTensor([input_token]).to(device)
            
            # Decode one step
            p_vocab, p_gen, hidden, cell, attention_weights, coverage = model.decoder(
                input_tensor, hidden, cell, encoder_outputs, mask, coverage
            )
            
            # Calculate final distribution
            vocab_size = len(vocab)
            oov_size = len(oov_words)
            extended_vocab_size = vocab_size + oov_size
            
            # Weighted vocabulary distribution
            p_vocab_weighted = p_gen * p_vocab
            
            # Weighted attention distribution
            p_copy_weighted = (1 - p_gen) * attention_weights
            
            # Create extended vocabulary distribution
            final_dist = torch.zeros(1, extended_vocab_size).to(device)
            final_dist[:, :vocab_size] = p_vocab_weighted
            final_dist.scatter_add_(1, src_extended_tensor, p_copy_weighted)
            
            # Get predicted token
            pred_token = final_dist.argmax(1).item()
            
            # Stop if EOS token is generated
            if pred_token == config.EOS_IDX:
                break
            
            output_tokens.append(pred_token)
            input_token = pred_token
        
        # Decode with OOV handling
        summary_words = vocab.decode_with_oov(output_tokens, oov_words)
        summary = ' '.join(summary_words)
    
    return summary


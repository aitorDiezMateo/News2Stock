"""
Pointer-Generator Network for text summarization

Based on "Get To The Point: Summarization with Pointer-Generator Networks"
by See et al. (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import config

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("Warning: fasttext not available. Install with: pip install fasttext")


class Encoder(nn.Module):
    """Bidirectional LSTM Encoder"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layers to reduce bidirectional hidden states
        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, src, src_lengths):
        """
        Args:
            src: [batch_size, src_len]
            src_lengths: [batch_size]
        Returns:
            encoder_outputs: [batch_size, src_len, hidden_dim * 2]
            hidden: [num_layers, batch_size, hidden_dim]
            cell: [num_layers, batch_size, hidden_dim]
        """
        # Embed and apply dropout
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Forward through LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # encoder_outputs: [batch_size, src_len, hidden_dim * 2]
        
        # Reduce bidirectional hidden states
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # We need to combine forward and backward for each layer
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)  # [num_layers, 2, batch_size, hidden_dim]
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        
        # Concatenate forward and backward
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # [num_layers, batch_size, hidden_dim * 2]
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        # Reduce to decoder hidden dimension
        hidden = self.reduce_h(hidden)  # [num_layers, batch_size, hidden_dim]
        cell = self.reduce_c(cell)
        
        return encoder_outputs, (hidden, cell)


class Attention(nn.Module):
    """Bahdanau attention with coverage mechanism"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, use_coverage=False):
        super(Attention, self).__init__()
        
        self.use_coverage = use_coverage
        
        # Attention layers
        self.W_h = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)  # encoder hidden
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)  # decoder hidden
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
        # Coverage layer
        if use_coverage:
            self.W_c = nn.Linear(1, decoder_hidden_dim, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, mask, coverage=None):
        """
        Args:
            decoder_hidden: [batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len]
            coverage: [batch_size, src_len] (optional, for coverage mechanism)
        Returns:
            attention_weights: [batch_size, src_len]
            coverage: [batch_size, src_len] (updated coverage)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Project encoder outputs
        encoder_features = self.W_h(encoder_outputs)  # [batch_size, src_len, decoder_hidden_dim]
        
        # Project decoder hidden state and expand
        decoder_features = self.W_s(decoder_hidden).unsqueeze(1)  # [batch_size, 1, decoder_hidden_dim]
        decoder_features = decoder_features.expand(-1, src_len, -1)  # [batch_size, src_len, decoder_hidden_dim]
        
        # Calculate attention scores
        att_features = encoder_features + decoder_features  # [batch_size, src_len, decoder_hidden_dim]
        
        # Add coverage features if enabled
        if self.use_coverage and coverage is not None:
            coverage_features = self.W_c(coverage.unsqueeze(2))  # [batch_size, src_len, decoder_hidden_dim]
            att_features = att_features + coverage_features
        
        # Calculate attention scores
        e = self.v(torch.tanh(att_features)).squeeze(2)  # [batch_size, src_len]
        
        # Mask padding positions
        e = e.masked_fill(mask == 0, -1e10)
        
        # Apply softmax
        attention_weights = F.softmax(e, dim=1)  # [batch_size, src_len]
        
        # Update coverage
        if self.use_coverage:
            if coverage is None:
                coverage = attention_weights
            else:
                coverage = coverage + attention_weights
        
        return attention_weights, coverage


class Decoder(nn.Module):
    """LSTM Decoder with attention and pointer-generator mechanism"""
    
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, 
                 num_layers, dropout, attention):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.attention = attention
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM input is embedding + context vector
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,
            decoder_hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layers for generation probability (pointer-generator switch)
        self.W_h = nn.Linear(encoder_hidden_dim, 1)  # context vector
        self.W_s = nn.Linear(decoder_hidden_dim, 1)  # decoder state
        self.W_x = nn.Linear(embedding_dim, 1)  # decoder input
        
        # Output layer for vocabulary distribution
        self.V = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.V2 = nn.Linear(decoder_hidden_dim, vocab_size)
        
    def forward(self, input, hidden, cell, encoder_outputs, mask, coverage=None):
        """
        Args:
            input: [batch_size]
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len]
            coverage: [batch_size, src_len] (optional)
        Returns:
            p_vocab: [batch_size, vocab_size] - vocabulary distribution
            p_gen: [batch_size, 1] - generation probability
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
            attention_weights: [batch_size, src_len]
            coverage: [batch_size, src_len]
        """
        # input: [batch_size] -> [batch_size, 1]
        input = input.unsqueeze(1)
        
        # Embed and dropout
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # Calculate attention weights using top layer hidden state
        attention_weights, coverage = self.attention(
            hidden[-1], encoder_outputs, mask, coverage
        )  # [batch_size, src_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, encoder_hidden_dim]
        
        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + encoder_hidden_dim]
        
        # Forward through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch_size, 1, decoder_hidden_dim]
        
        # Calculate generation probability (pointer-generator switch)
        output_squeezed = output.squeeze(1)  # [batch_size, decoder_hidden_dim]
        context_squeezed = context.squeeze(1)  # [batch_size, encoder_hidden_dim]
        embedded_squeezed = embedded.squeeze(1)  # [batch_size, emb_dim]
        
        p_gen = torch.sigmoid(
            self.W_h(context_squeezed) + 
            self.W_s(output_squeezed) + 
            self.W_x(embedded_squeezed)
        )  # [batch_size, 1]
        
        # Calculate vocabulary distribution
        concat = torch.cat((output_squeezed, context_squeezed), dim=1)  # [batch_size, decoder_hidden_dim + encoder_hidden_dim]
        p_vocab = self.V2(torch.relu(self.V(concat)))  # [batch_size, vocab_size]
        p_vocab = F.softmax(p_vocab, dim=1)
        
        return p_vocab, p_gen, hidden, cell, attention_weights, coverage


class PointerGeneratorNetwork(nn.Module):
    """Complete Pointer-Generator Network"""
    
    def __init__(self, encoder, decoder, device):
        super(PointerGeneratorNetwork, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_lengths, trg, src_extended, oov_size, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch_size, src_len] - source with vocab indices
            src_lengths: [batch_size]
            trg: [batch_size, trg_len] - target with vocab indices
            src_extended: [batch_size, src_len] - source with extended vocab (including OOV)
            oov_size: int - number of OOV words in this batch
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            final_dists: [batch_size, trg_len, vocab_size + oov_size]
            coverages: list of [batch_size, src_len] (for coverage loss)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.vocab_size
        extended_vocab_size = vocab_size + oov_size
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Create mask for attention (1 for real tokens, 0 for padding)
        mask = (src != config.PAD_IDX)
        
        # Initialize coverage vector
        coverage = None
        coverages = []
        
        # Tensor to store final distributions
        final_dists = torch.zeros(batch_size, trg_len, extended_vocab_size).to(self.device)
        
        # First input to decoder is <SOS> token
        input = trg[:, 0]
        
        # Decode
        for t in range(1, trg_len):
            # Forward through decoder
            p_vocab, p_gen, hidden, cell, attention_weights, coverage = self.decoder(
                input, hidden, cell, encoder_outputs, mask, coverage
            )
            
            # Store coverage for loss calculation
            if config.USE_COVERAGE:
                coverages.append(coverage)
            
            # Calculate final distribution (pointer-generator mechanism)
            # p_vocab: [batch_size, vocab_size]
            # p_gen: [batch_size, 1]
            # attention_weights: [batch_size, src_len]
            
            # Weighted vocabulary distribution
            p_vocab_weighted = p_gen * p_vocab  # [batch_size, vocab_size]
            
            # Weighted attention distribution (copy distribution)
            p_copy_weighted = (1 - p_gen) * attention_weights  # [batch_size, src_len]
            
            # Create extended vocabulary distribution
            final_dist = torch.zeros(batch_size, extended_vocab_size).to(self.device)
            
            # Add vocabulary distribution
            final_dist[:, :vocab_size] = p_vocab_weighted
            
            # Add copy distribution (scatter attention weights to extended vocab positions)
            final_dist.scatter_add_(1, src_extended, p_copy_weighted)
            
            # Store final distribution
            final_dists[:, t, :] = final_dist
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token from vocabulary only
            # (we can't use extended vocab indices as input to decoder)
            top1 = p_vocab.argmax(1)
            
            # Use teacher forcing or predicted token as next input
            input = trg[:, t] if teacher_force else top1
        
        return final_dists, coverages


def calculate_loss(final_dists, target, coverages, attention_weights_list, padding_mask):
    """
    Calculate loss for pointer-generator network
    
    Args:
        final_dists: [batch_size, trg_len, extended_vocab_size]
        target: [batch_size, trg_len] - target with extended vocab indices
        coverages: list of [batch_size, src_len]
        attention_weights_list: list of [batch_size, src_len]
        padding_mask: [batch_size, trg_len] - 1 for real tokens, 0 for padding
    Returns:
        loss: scalar
    """
    # Negative log likelihood loss
    batch_size = final_dists.shape[0]
    trg_len = final_dists.shape[1]
    
    # Gather probabilities of target tokens
    # Reshape for gathering
    final_dists = final_dists.view(-1, final_dists.shape[-1])  # [batch_size * trg_len, extended_vocab_size]
    target = target.view(-1)  # [batch_size * trg_len]
    
    # Gather target probabilities
    target_probs = torch.gather(final_dists, 1, target.unsqueeze(1)).squeeze(1)  # [batch_size * trg_len]
    
    # Avoid log(0)
    target_probs = target_probs + 1e-10
    
    # Calculate negative log likelihood
    nll_loss = -torch.log(target_probs)  # [batch_size * trg_len]
    
    # Reshape and apply mask
    nll_loss = nll_loss.view(batch_size, trg_len)  # [batch_size, trg_len]
    padding_mask = padding_mask.view(batch_size, trg_len)
    
    # Mask padding positions
    nll_loss = nll_loss * padding_mask
    
    # Average over non-padding tokens
    nll_loss = nll_loss.sum() / padding_mask.sum()
    
    # Coverage loss
    coverage_loss = 0
    if config.USE_COVERAGE and len(coverages) > 0:
        # Coverage loss penalizes attention overlap
        for i, (coverage, attention) in enumerate(zip(coverages, attention_weights_list)):
            # coverage: [batch_size, src_len]
            # attention: [batch_size, src_len]
            # Take minimum of coverage and attention at each position
            step_coverage_loss = torch.sum(torch.min(coverage, attention), dim=1)  # [batch_size]
            
            # Apply target mask for this timestep
            if i + 1 < trg_len:
                step_mask = padding_mask[:, i + 1]
                step_coverage_loss = step_coverage_loss * step_mask
                coverage_loss += step_coverage_loss.sum()
        
        # Normalize by number of non-padding tokens
        coverage_loss = coverage_loss / padding_mask.sum()
        coverage_loss = config.COVERAGE_LAMBDA * coverage_loss
    
    # Total loss
    total_loss = nll_loss + coverage_loss
    
    return total_loss, nll_loss, coverage_loss


def init_embeddings_with_fasttext(embedding_layer, vocab, fasttext_model_path, embedding_dim):
    """
    Initialize embedding layer with FastText vectors
    
    Args:
        embedding_layer: nn.Embedding layer to initialize
        vocab: Vocabulary object with word2idx mapping
        fasttext_model_path: Path to FastText model file (.bin)
        embedding_dim: Dimension of embeddings (should match FastText model)
    """
    if not FASTTEXT_AVAILABLE:
        raise ImportError("fasttext library not available. Install with: pip install fasttext")
    
    print(f"Loading FastText model from {fasttext_model_path}...")
    ft_model = fasttext.load_model(fasttext_model_path)
    print(f"✓ FastText model loaded")
    
    # Initialize embedding matrix
    vocab_size = len(vocab)
    initialized_count = 0
    
    print("Initializing embeddings with FastText vectors...")
    with torch.no_grad():
        for word, idx in vocab.word2idx.items():
            if idx == config.PAD_IDX:
                # Keep padding token as zeros
                embedding_layer.weight.data[idx].zero_()
            elif word in [config.UNK_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN]:
                # Get FastText vector for special tokens
                vector = ft_model.get_word_vector(word)
                embedding_layer.weight.data[idx] = torch.from_numpy(vector[:embedding_dim])
                initialized_count += 1
            else:
                # Get FastText vector for regular words
                vector = ft_model.get_word_vector(word)
                embedding_layer.weight.data[idx] = torch.from_numpy(vector[:embedding_dim])
                initialized_count += 1
    
    print(f"✓ Initialized {initialized_count}/{vocab_size} embeddings with FastText vectors")
    
    # Free up memory
    del ft_model
    
    return embedding_layer


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
    """LSTM Encoder with optional bidirectionality"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, bidirectional=True):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
    def forward(self, src, src_lengths):
        """
        Args:
            src: [batch_size, src_len]
            src_lengths: [batch_size]
        Returns:
            outputs: [batch_size, src_len, hidden_dim * num_directions]
            hidden: tuple of (h, c) each [num_layers * num_directions, batch_size, hidden_dim]
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
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # If bidirectional, concatenate forward and backward hidden states
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # We need to reshape to: [num_layers, batch_size, hidden_dim * num_directions]
        if self.bidirectional:
            # Reshape hidden and cell states
            hidden = self._combine_bidirectional(hidden)
            cell = self._combine_bidirectional(cell)
        
        return outputs, (hidden, cell)
    
    def _combine_bidirectional(self, state):
        """
        Combine bidirectional hidden states
        state: [num_layers * 2, batch_size, hidden_dim]
        returns: [num_layers, batch_size, hidden_dim * 2]
        """
        # Reshape from [num_layers * 2, batch_size, hidden_dim]
        # to [num_layers, 2, batch_size, hidden_dim]
        state = state.view(self.num_layers, 2, -1, self.hidden_dim)
        # Concatenate forward and backward
        state = torch.cat([state[:, 0, :, :], state[:, 1, :, :]], dim=2)
        # Result: [num_layers, batch_size, hidden_dim * 2]
        return state


class Attention(nn.Module):
    """Bahdanau attention mechanism"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask):
        """
        Args:
            hidden: [batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len]
        Returns:
            attention_weights: [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, decoder_hidden_dim]
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Mask attention scores for padded positions
        attention = attention.masked_fill(mask == 0, -1e10)
        
        # Apply softmax
        attention_weights = F.softmax(attention, dim=1)
        
        return attention_weights


class Decoder(nn.Module):
    """LSTM Decoder with attention"""
    
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
        
        # Output layer
        self.fc_out = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, vocab_size)
        
    def forward(self, input, hidden, cell, encoder_outputs, mask):
        """
        Args:
            input: [batch_size]
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len]
        Returns:
            prediction: [batch_size, vocab_size]
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
            attention_weights: [batch_size, src_len]
        """
        # input: [batch_size] -> [batch_size, 1]
        input = input.unsqueeze(1)
        
        # Embed and dropout
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # Calculate attention weights using top layer hidden state
        attention_weights = self.attention(hidden[-1], encoder_outputs, mask)  # [batch_size, src_len]
        
        # Calculate context vector
        attention_weights_unsqueezed = attention_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(attention_weights_unsqueezed, encoder_outputs)  # [batch_size, 1, encoder_hidden_dim]
        
        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + encoder_hidden_dim]
        
        # Forward through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch_size, 1, decoder_hidden_dim]
        
        # Concatenate output, context, and embedding for prediction
        output = output.squeeze(1)  # [batch_size, decoder_hidden_dim]
        context = context.squeeze(1)  # [batch_size, encoder_hidden_dim]
        embedded = embedded.squeeze(1)  # [batch_size, emb_dim]
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch_size, vocab_size]
        
        return prediction, hidden, cell, attention_weights


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model"""
    
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Linear layer to project encoder hidden state to decoder hidden state dimension
        encoder_hidden_dim = encoder.hidden_dim * encoder.num_directions
        decoder_hidden_dim = decoder.lstm.hidden_size
        
        if encoder_hidden_dim != decoder_hidden_dim:
            self.bridge_h = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
            self.bridge_c = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        else:
            self.bridge_h = None
            self.bridge_c = None
        
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch_size, src_len]
            src_lengths: [batch_size]
            trg: [batch_size, trg_len]
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: [batch_size, trg_len, vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Project encoder hidden state to decoder hidden state dimension if needed
        if self.bridge_h is not None:
            hidden = self.bridge_h(hidden)
            cell = self.bridge_c(cell)
        
        # Create mask for attention (1 for real tokens, 0 for padding)
        mask = (src != config.PAD_IDX)
        
        # First input to decoder is <SOS> token
        input = trg[:, 0]
        
        # Decode
        for t in range(1, trg_len):
            # Forward through decoder
            output, hidden, cell, attention_weights = self.decoder(
                input, hidden, cell, encoder_outputs, mask
            )
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use teacher forcing or predicted token as next input
            input = trg[:, t] if teacher_force else top1
        
        return outputs


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
    
    print(f"Initialized {initialized_count}/{vocab_size} embeddings with FastText vectors")
    
    # Free up memory
    del ft_model
    
    return embedding_layer


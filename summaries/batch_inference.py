"""
Batch inference script for generating summaries with all four trained models
"""

import os
import sys
import re
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from datetime import datetime


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize(text: str) -> list:
    """Simple word tokenization"""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def load_seq2seq_model(model_dir, device):
    """Load simple seq2seq model"""
    # Clean any previous imports to avoid conflicts
    modules_to_remove = ['config', 'model', 'vocabulary', 'dataset']
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]
    
    # Temporarily add model directory to path
    sys.path.insert(0, model_dir)
    
    try:
        # Import modules (they will now find 'config' correctly)
        import config
        from model import Encoder, Decoder, Attention, Seq2Seq
        
        # Load vocabulary
        vocab_path = os.path.join(model_dir, 'preprocessed_data.pkl')
        with open(vocab_path, 'rb') as f:
            data_dict = pickle.load(f)
        vocab = data_dict['vocab']
        
        # Initialize model
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
        
        # Load weights
        checkpoint_path = os.path.join(model_dir, 'checkpoints', 'best_model.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, vocab, config
    
    finally:
        # Remove from path
        sys.path.pop(0)


def load_pointer_generator_model(model_dir, device):
    """Load pointer-generator model"""
    # Clean any previous imports to avoid conflicts
    modules_to_remove = ['config', 'model', 'vocabulary', 'dataset']
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]
    
    # Temporarily add model directory to path
    sys.path.insert(0, model_dir)
    
    try:
        # Import modules
        import config
        from model import Encoder, Decoder, Attention, PointerGeneratorNetwork
        
        # Load vocabulary
        vocab_path = os.path.join(model_dir, 'preprocessed_data.pkl')
        with open(vocab_path, 'rb') as f:
            data_dict = pickle.load(f)
        vocab = data_dict['vocab']
        
        # Initialize model
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
        checkpoint_path = os.path.join(model_dir, 'checkpoints', 'best_model.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, vocab, config
    
    finally:
        # Remove from path
        sys.path.pop(0)


def generate_summary_seq2seq(model, source_text, vocab, config, device, max_length=100):
    """Generate summary using simple seq2seq model"""
    model.eval()
    
    # Preprocess
    cleaned = clean_text(source_text)
    tokens = tokenize(cleaned)
    src_indices = vocab.encode(tokens)
    src_indices = src_indices[:config.MAX_SOURCE_LEN]
    
    # Convert to tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
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
            trg_indices.append(pred_token)
            
            # Stop if EOS token is generated
            if pred_token == config.EOS_IDX:
                break
        
        # Decode indices to words
        summary_tokens = vocab.decode(trg_indices[1:-1])  # Exclude SOS and EOS
        summary = ' '.join(summary_tokens)
    
    return summary


def generate_summary_pointer_generator(model, source_text, vocab, config, device, max_length=100):
    """Generate summary using pointer-generator model (greedy decoding)"""
    model.eval()
    
    # Preprocess
    cleaned = clean_text(source_text)
    tokens = tokenize(cleaned)
    
    # Encode with OOV handling
    src_extended, oov_words = vocab.encode_with_oov(tokens)
    src = vocab.encode(tokens)
    
    # Truncate if too long
    src = src[:config.MAX_SOURCE_LEN]
    src_extended = src_extended[:config.MAX_SOURCE_LEN]
    
    # Convert to tensors
    src_tensor = torch.LongTensor(src).unsqueeze(0).to(device)
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
        for _ in range(max_length):
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


def main():
    """Main batch inference function"""
    print("="*80)
    print("BATCH INFERENCE FOR ALL MODELS")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # summaries folder
    root_dir = os.path.dirname(script_dir)  # News2Stock folder
    
    # Define model configurations with absolute paths
    models_config = {
        'simple_seq2seq': {
            'dir': os.path.join(script_dir, 'simple_seq2seq'),
            'type': 'seq2seq',
            'output': os.path.join(root_dir, 'data/news/inference/apple_news_simple_seq2seq.parquet')
        },
        'simple_seq2seq_fasttext': {
            'dir': os.path.join(script_dir, 'simple_seq2seq_fasttext'),
            'type': 'seq2seq',
            'output': os.path.join(root_dir, 'data/news/inference/apple_news_simple_seq2seq_fasttext.parquet')
        },
        'pointer_generator': {
            'dir': os.path.join(script_dir, 'pointer_generator'),
            'type': 'pointer_generator',
            'output': os.path.join(root_dir, 'data/news/inference/apple_news_pointer_generator.parquet')
        },
        'pointer_generator_fasttext': {
            'dir': os.path.join(script_dir, 'pointer_generator_fasttext'),
            'type': 'pointer_generator',
            'output': os.path.join(root_dir, 'data/news/inference/apple_news_pointer_generator_fasttext.parquet')
        }
    }
    
    # Create output directory
    output_dir = os.path.join(root_dir, 'data/news/inference')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Apple news data
    print("\nLoading Apple news data...")
    data_path = os.path.join(root_dir, 'data/news/summarized/apple_news.parquet')
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df)} articles")
    
    # Process each model
    for model_name, model_config in models_config.items():
        print("\n" + "="*80)
        print(f"Processing with {model_name}")
        print("="*80)
        
        model_dir = model_config['dir']
        model_type = model_config['type']
        output_path = model_config['output']
        
        # Load model
        print(f"\nLoading model from {model_dir}...")
        try:
            if model_type == 'seq2seq':
                model, vocab, config = load_seq2seq_model(model_dir, device)
            else:  # pointer_generator
                model, vocab, config = load_pointer_generator_model(model_dir, device)
            
            print(f"✓ Model loaded successfully")
            print(f"  Vocabulary size: {len(vocab)}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            continue
        
        # Filter out empty articles and articles that produce empty tokens
        print(f"\nFiltering empty articles...")
        valid_indices = []
        for idx, row in df.iterrows():
            text = row['clean_body']
            if text is not None and text != '':
                # Check if text produces valid tokens after cleaning
                cleaned = clean_text(text)
                tokens = tokenize(cleaned)
                encoded = vocab.encode(tokens)
                if len(encoded) > 0:
                    valid_indices.append(idx)
        
        df_filtered = df.loc[valid_indices].copy()
        print(f"Generating summaries for non-empty articles: {len(df_filtered)}/{len(df)} articles")
        
        summaries = []
        
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Inference {model_name}"):
            source_text = row['clean_body']
            
            try:
                if model_type == 'seq2seq':
                    summary = generate_summary_seq2seq(
                        model, source_text, vocab, config, device
                    )
                else:  # pointer_generator
                    summary = generate_summary_pointer_generator(
                        model, source_text, vocab, config, device
                    )
                summaries.append(summary)
            except Exception as e:
                print(f"\n✗ Error on article {idx}: {e}")
                summaries.append("")  # Empty summary on error
        
        # Create output dataframe (only non-empty articles)
        output_df = df_filtered.copy()
        output_df['generated_summary'] = summaries
        output_df['model'] = model_name
        output_df['inference_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to parquet
        output_df.to_parquet(output_path, index=False)
        print(f"\n✓ Saved summaries to {output_path}")
        print(f"  Total articles: {len(output_df)}")
        print(f"  Columns: {list(output_df.columns)}")
    
    print("\n" + "="*80)
    print("BATCH INFERENCE COMPLETED")
    print("="*80)
    print("\nGenerated files:")
    for model_name, model_config in models_config.items():
        output_path = model_config['output']
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ {output_path} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()


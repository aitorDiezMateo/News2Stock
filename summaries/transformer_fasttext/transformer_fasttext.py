import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import numpy as np
from typing import Tuple, Optional, Dict, List
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                     V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(-2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output, attn_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)
        
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm2(x + cross_attn_output)
        
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm3(x + ff_output)
        
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) 
                                     for _ in range(num_layers)])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) 
                                     for _ in range(num_layers)])
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class TransformerSummarizer(nn.Module):
    def __init__(self, embedding_dim: int = 300, d_model: int = 256, num_heads: int = 8, 
                 d_ff: int = 1024, num_encoder_layers: int = 4, num_decoder_layers: int = 4, 
                 dropout: float = 0.1, max_seq_len: int = 1024, vocab_size: int = 10000):
        super(TransformerSummarizer, self).__init__()
        
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        
        self.src_projection = nn.Linear(embedding_dim, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        self.src_positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.tgt_positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.encoder = Encoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_decoder_layers, dropout)
        
        self.output_linear = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_mask = torch.ones(1, 1, 1, src.size(1), device=src.device).float()
        
        seq_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).unsqueeze(0).unsqueeze(0)
        
        return src_mask, tgt_mask.float()
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = self.create_mask(src, tgt)
        
        src_proj = self.src_projection(src)
        src_proj = self.src_positional_encoding(src_proj)
        encoder_output = self.encoder(src_proj, src_mask)
        
        tgt_embed = self.tgt_embedding(tgt)
        tgt_embed = self.tgt_positional_encoding(tgt_embed)
        decoder_output = self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask)
        
        output = self.output_linear(decoder_output)
        return output


class TransformerSummarizerTrainer:
    def __init__(self, model: TransformerSummarizer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for src, tgt, tgt_y in pbar:
            src = src.to(self.device).float()
            tgt = tgt.to(self.device).long()
            tgt_y = tgt_y.to(self.device).long()
            
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt)
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_y.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def val_epoch(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        with torch.no_grad():
            for src, tgt, tgt_y in pbar:
                src = src.to(self.device).float()
                tgt = tgt.to(self.device).long()
                tgt_y = tgt_y.to(self.device).long()
                
                output = self.model(src, tgt)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_y.reshape(-1))
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def fit(self, train_dataloader, val_dataloader, epochs: int = 50, early_stopping_patience: int = 10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar:
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.val_epoch(val_dataloader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model_fasttext.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    pbar.close()
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def generate_summary(self, src: torch.Tensor, max_len: int = 100, 
                        start_token_id: int = 1, end_token_id: int = 2) -> torch.Tensor:
        self.model.eval()
        
        src = src.unsqueeze(0).to(self.device).float()
        
        src_mask = torch.ones(1, 1, src.size(1), device=self.device)
        src_proj = self.model.src_projection(src)
        src_proj = self.model.src_positional_encoding(src_proj)
        encoder_output = self.model.encoder(src_proj, src_mask)
        
        tgt = torch.tensor([[start_token_id]], device=self.device)
        
        with torch.no_grad():
            for _ in range(max_len):
                tgt_mask = torch.ones(1, 1, tgt.size(1), tgt.size(1), device=self.device)
                tgt_mask = torch.tril(tgt_mask)
                
                tgt_embed = self.model.tgt_embedding(tgt)
                tgt_embed = self.model.tgt_positional_encoding(tgt_embed)
                decoder_output = self.model.decoder(tgt_embed, encoder_output, src_mask, tgt_mask)
                
                output = self.model.output_linear(decoder_output)
                next_token = output[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if next_token.item() == end_token_id:
                    break
        
        return tgt.squeeze(0)
    
    def save_checkpoint(self, filepath: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']


class FastTextEmbeddingDataset:
    def __init__(self, texts: List[str], summaries: List[str], embeddings_source, 
                 tokenizer, max_src_len: int = 512, max_tgt_len: int = 128, embedding_dim: int = 300):
        self.texts = texts
        self.summaries = summaries
        self.embeddings_source = embeddings_source  # Can be dict-like or FastTextWordEmbeddings
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.embedding_dim = embedding_dim
    
    def __len__(self):
        return len(self.texts)
    
    def _get_word_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word, returns zero vector if not found"""
        try:
            if word in self.embeddings_source:
                emb = self.embeddings_source[word]
                if isinstance(emb, np.ndarray):
                    return emb
                return np.array(emb)
        except:
            pass
        return np.zeros(self.embedding_dim)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        words = text.lower().split()[:self.max_src_len]
        src_embeddings = [self._get_word_embedding(word) for word in words]
        
        if not src_embeddings:
            src_embeddings = [np.zeros(self.embedding_dim)]
        
        src_embeddings = np.array(src_embeddings)
        pad_len = self.max_src_len - len(src_embeddings)
        if pad_len > 0:
            src_embeddings = np.vstack([src_embeddings, np.zeros((pad_len, self.embedding_dim))])
        elif pad_len < 0:
            src_embeddings = src_embeddings[:self.max_src_len]
        
        src = torch.tensor(src_embeddings, dtype=torch.float32)
        
        tgt_tokens = self.tokenizer.encode(summary)[:self.max_tgt_len - 1]
        tgt_pad_len = self.max_tgt_len - len(tgt_tokens) - 1
        
        tgt = torch.tensor([1] + tgt_tokens + [0] * tgt_pad_len, dtype=torch.long)
        tgt_y = torch.tensor(tgt_tokens + [2] + [0] * tgt_pad_len, dtype=torch.long)
        
        return src, tgt, tgt_y


class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = {}
    
    def build_vocab(self, texts):
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_count[word] = self.word_count.get(word, 0) + 1
        
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.vocab_size - 4], start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.word2idx['<PAD>'] = 0
        self.word2idx['<START>'] = 1
        self.word2idx['<END>'] = 2
        self.word2idx['<UNK>'] = 3
    
    def encode(self, text):
        words = text.lower().split()
        return [self.word2idx.get(word, 3) for word in words]
    
    def decode(self, tokens):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in tokens if idx > 3])


class FastTextWordEmbeddings:
    """Wrapper to get word embeddings from FastText model"""
    def __init__(self, model_path: str):
        import fasttext
        print(f"Loading FastText model from {model_path}...")
        self.model = fasttext.load_model(model_path)
        print(f"FastText model loaded. Dimension: {self.model.get_dimension()}")
    
    def __contains__(self, word: str) -> bool:
        # FastText can generate embeddings for any word (including OOV via subwords)
        return True
    
    def __getitem__(self, word: str) -> np.ndarray:
        return self.model.get_word_vector(word)
    
    def get(self, word: str, default=None) -> np.ndarray:
        return self.model.get_word_vector(word)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print("Loading all news data (excluding Apple)...")
    all_data = []
    news_dir = Path(__file__).parent.parent.parent / "data" / "news" / "summarized"
    
    print(f"Looking for files in: {news_dir}")
    for parquet_file in sorted(news_dir.glob("*_news.parquet")):
        ticker = parquet_file.stem.split("_")[0]
        if ticker.upper() != "APPLE":
            df = pd.read_parquet(parquet_file)
            df = df[['clean_body', 'body_summary']].dropna()
            all_data.append(df)
            print(f"  Loaded {ticker}: {len(df)} articles")
    
    if not all_data:
        raise ValueError(f"No parquet files found in {news_dir}")
    
    all_train_data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data (excluding Apple): {len(all_train_data)} articles")
    
    # Split 80/20 for training and validation
    train_data, val_data = train_test_split(all_train_data, test_size=0.2, random_state=42)
    print(f"Training data: {len(train_data)} articles (80%)")
    print(f"Validation data: {len(val_data)} articles (20%)")
    
    print("\nLoading Apple news for testing...")
    test_data = pd.read_parquet(news_dir / "apple_news.parquet")
    test_data = test_data[['clean_body', 'body_summary']].dropna()
    print(f"Total test data (Apple): {len(test_data)} articles")
    
    print("\nLoading FastText model for word embeddings...")
    fasttext_model_path = Path(__file__).parent.parent.parent / "models" / "cc.en.300.bin"
    
    if not fasttext_model_path.exists():
        raise FileNotFoundError(
            f"FastText model not found at {fasttext_model_path}. "
            "Please download cc.en.300.bin from https://fasttext.cc/docs/en/crawl-vectors.html"
        )
    
    # Use the FastText model directly for word embeddings
    embeddings_dict = FastTextWordEmbeddings(str(fasttext_model_path))
    print("FastText word embeddings ready!")
    
    tokenizer = SimpleTokenizer(vocab_size=10000)
    tokenizer.build_vocab(pd.concat([all_train_data['clean_body'], all_train_data['body_summary']]).tolist())
    
    train_dataset = FastTextEmbeddingDataset(
        train_data['clean_body'].tolist(),
        train_data['body_summary'].tolist(),
        embeddings_dict,
        tokenizer,
        max_src_len=512,
        max_tgt_len=128
    )
    
    val_dataset = FastTextEmbeddingDataset(
        val_data['clean_body'].tolist(),
        val_data['body_summary'].tolist(),
        embeddings_dict,
        tokenizer,
        max_src_len=512,
        max_tgt_len=128
    )
    
    test_dataset = FastTextEmbeddingDataset(
        test_data['clean_body'].tolist(),
        test_data['body_summary'].tolist(),
        embeddings_dict,
        tokenizer,
        max_src_len=512,
        max_tgt_len=128
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = TransformerSummarizer(
        embedding_dim=300,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.1,
        max_seq_len=512,
        vocab_size=10000
    )
    
    trainer = TransformerSummarizerTrainer(model, device=device)
    trainer.fit(train_loader, val_loader, epochs=20, early_stopping_patience=5)
    
    print("\n" + "="*80)
    print("TESTING MODEL ON APPLE NEWS")
    print("="*80)
    
    trainer.load_checkpoint('best_model_fasttext.pt')
    
    test_loss = trainer.val_epoch(test_loader)
    print(f"\nTest Loss (Apple): {test_loss:.4f}")
    
    print("\n" + "-"*80)
    print("GENERATING SUMMARIES FOR APPLE NEWS")
    print("-"*80)
    
    from datetime import datetime
    generated_data = []
    inference_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_word_embedding(word: str, emb_source, dim: int = 300) -> np.ndarray:
        """Get embedding for a word from the FastText model"""
        try:
            if word in emb_source:
                emb = emb_source[word]
                return np.array(emb) if not isinstance(emb, np.ndarray) else emb
        except:
            pass
        return np.zeros(dim)
    
    for i in range(len(test_data)):
        src_text = test_data.iloc[i]['clean_body']
        
        words = src_text.lower().split()[:512]
        src_embeddings = [get_word_embedding(word, embeddings_dict) for word in words]
        
        if not src_embeddings:
            src_embeddings = [np.zeros(300)]
        
        src_embeddings = np.array(src_embeddings)
        pad_len = 512 - len(src_embeddings)
        if pad_len > 0:
            src_embeddings = np.vstack([src_embeddings, np.zeros((pad_len, 300))])
        elif pad_len < 0:
            src_embeddings = src_embeddings[:512]
        
        src_tensor = torch.tensor(src_embeddings, dtype=torch.float32)
        
        generated_tokens = trainer.generate_summary(src_tensor, max_len=128)
        generated_summary = tokenizer.decode(generated_tokens.cpu().numpy())
        
        row_dict = {
            'id': test_data.iloc[i].get('id', i),
            'author': test_data.iloc[i].get('author', ''),
            'created': test_data.iloc[i].get('created', ''),
            'title': test_data.iloc[i].get('title', ''),
            'teaser': test_data.iloc[i].get('teaser', ''),
            'clean_body': src_text,
            'body_summary': test_data.iloc[i]['body_summary'],
            'generated_summary': generated_summary,
            'model': 'transformer_fasttext',
            'inference_date': inference_date
        }
        generated_data.append(row_dict)
        
        if i < 5:
            print(f"\n[Sample {i+1}]")
            print(f"Original Text: {src_text[:150]}...")
            print(f"True Summary: {test_data.iloc[i]['body_summary']}")
            print(f"Generated Summary: {generated_summary}")
    
    output_df = pd.DataFrame(generated_data)
    output_path = Path(__file__).parent.parent.parent / "data" / "news" / "inference" / "apple_news_transformer_fasttext.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path)
    
    print(f"\n✓ Saved {len(output_df)} generated summaries to {output_path}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

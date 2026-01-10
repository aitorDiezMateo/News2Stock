"""
Sentiment Analysis Module
==========================
Uses pre-trained models for financial sentiment analysis.
No labeled data required - uses FinBERT (fine-tuned on financial text).

Available models:
- FinBERT: Best for financial news (requires transformers)
- VADER: Lexicon-based, fast, no GPU needed
- TextBlob: Simple polarity analysis
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Base class for sentiment analyzers."""
    
    def __init__(self):
        self.name = "base"
        
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            Dict with 'positive', 'negative', 'neutral' scores and 'compound'
        """
        raise NotImplementedError
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze(text) for text in tqdm(texts, desc=f"Analyzing sentiment ({self.name})")]


class FinBERTSentiment(SentimentAnalyzer):
    """
    FinBERT-based sentiment analyzer.
    Pre-trained on financial news, best accuracy for stock-related text.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.name = "FinBERT"
        self.device = device
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            print("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.to(device)
            self.model.eval()
            self.torch = torch
            print("✓ FinBERT loaded successfully")
            
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load FinBERT: {e}")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze single text with FinBERT."""
        if not text or len(text.strip()) == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        # Truncate long texts
        text = text[:512]
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probs = self.torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        # FinBERT outputs: [positive, negative, neutral]
        return {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
            'compound': float(probs[0] - probs[1])  # Range: -1 to 1
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Batch analysis for efficiency."""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT sentiment"):
            batch_texts = texts[i:i+batch_size]
            
            # Filter empty texts
            valid_texts = [t[:512] if t and len(t.strip()) > 0 else "neutral" for t in batch_texts]
            
            inputs = self.tokenizer(
                valid_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                probs = self.torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            for j, prob in enumerate(probs):
                results.append({
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2]),
                    'compound': float(prob[0] - prob[1])
                })
        
        return results


class VADERSentiment(SentimentAnalyzer):
    """
    VADER sentiment analyzer.
    Fast, lexicon-based, good for social media and news.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "VADER"
        
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            print("✓ VADER loaded successfully")
        except ImportError:
            raise ImportError("Please install vaderSentiment: pip install vaderSentiment")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze single text with VADER."""
        if not text or len(text.strip()) == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        scores = self.analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }


class TextBlobSentiment(SentimentAnalyzer):
    """
    TextBlob sentiment analyzer.
    Simple polarity-based analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "TextBlob"
        
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
            print("✓ TextBlob loaded successfully")
        except ImportError:
            raise ImportError("Please install textblob: pip install textblob")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze single text with TextBlob."""
        if not text or len(text.strip()) == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        blob = self.TextBlob(text)
        polarity = blob.sentiment.polarity  # Range: -1 to 1
        
        # Convert polarity to positive/negative/neutral
        if polarity > 0.1:
            return {
                'positive': polarity,
                'negative': 0.0,
                'neutral': 1 - polarity,
                'compound': polarity
            }
        elif polarity < -0.1:
            return {
                'positive': 0.0,
                'negative': -polarity,
                'neutral': 1 + polarity,
                'compound': polarity
            }
        else:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': polarity
            }


def get_sentiment_analyzer(model_type: str = 'finbert', device: str = 'cuda') -> SentimentAnalyzer:
    """
    Factory function to get sentiment analyzer.
    
    Args:
        model_type: 'finbert', 'vader', or 'textblob'
        device: 'cuda' or 'cpu' (for FinBERT)
        
    Returns:
        SentimentAnalyzer instance
    """
    if model_type.lower() == 'finbert':
        return FinBERTSentiment(device=device)
    elif model_type.lower() == 'vader':
        return VADERSentiment()
    elif model_type.lower() == 'textblob':
        return TextBlobSentiment()
    else:
        raise ValueError(f"Unknown sentiment model: {model_type}. Choose from: finbert, vader, textblob")


def compute_sentiment_features(
    news_df: pd.DataFrame,
    end_date: pd.Timestamp,
    window_days: int = 20,
    analyzer: Optional[SentimentAnalyzer] = None,
    text_column: str = 'title'
) -> Optional[np.ndarray]:
    """
    Compute sentiment features for news in a time window.
    
    Args:
        news_df: DataFrame with news data (must have 'Date' and text_column)
        end_date: End of the time window
        window_days: Days to look back
        analyzer: SentimentAnalyzer instance (if None, uses cached sentiments)
        text_column: Column containing text to analyze
        
    Returns:
        Array of sentiment features:
        [mean_compound, std_compound, mean_positive, mean_negative, 
         pct_positive, pct_negative, sentiment_trend, news_volume]
    """
    from datetime import timedelta
    
    start_date = end_date - timedelta(days=window_days)
    mask = (news_df['Date'] >= start_date) & (news_df['Date'] <= end_date)
    window_news = news_df[mask]
    
    if len(window_news) == 0:
        return None
    
    # Check if sentiment already computed
    if 'sentiment_compound' not in window_news.columns:
        if analyzer is None:
            return None
        
        # Compute sentiment on the fly (slow)
        texts = window_news[text_column].fillna('').tolist()
        sentiments = analyzer.analyze_batch(texts)
        compounds = [s['compound'] for s in sentiments]
        positives = [s['positive'] for s in sentiments]
        negatives = [s['negative'] for s in sentiments]
    else:
        compounds = window_news['sentiment_compound'].values
        positives = window_news['sentiment_positive'].values
        negatives = window_news['sentiment_negative'].values
    
    # Compute aggregated features
    mean_compound = np.mean(compounds)
    std_compound = np.std(compounds) if len(compounds) > 1 else 0.0
    mean_positive = np.mean(positives)
    mean_negative = np.mean(negatives)
    
    # Percentage of positive/negative news
    pct_positive = np.mean([1 if c > 0.1 else 0 for c in compounds])
    pct_negative = np.mean([1 if c < -0.1 else 0 for c in compounds])
    
    # Sentiment trend (recent vs older)
    if len(compounds) >= 4:
        mid = len(compounds) // 2
        recent_sentiment = np.mean(compounds[mid:])
        older_sentiment = np.mean(compounds[:mid])
        sentiment_trend = recent_sentiment - older_sentiment
    else:
        sentiment_trend = 0.0
    
    # News volume (normalized)
    news_volume = np.log1p(len(window_news))
    
    features = np.array([
        mean_compound,
        std_compound,
        mean_positive,
        mean_negative,
        pct_positive,
        pct_negative,
        sentiment_trend,
        news_volume
    ], dtype=np.float32)
    
    return features


# Sentiment feature names for reference
SENTIMENT_FEATURE_NAMES = [
    'sentiment_mean',
    'sentiment_std',
    'positive_mean',
    'negative_mean',
    'pct_positive',
    'pct_negative',
    'sentiment_trend',
    'news_volume'
]

SENTIMENT_FEATURES_DIM = len(SENTIMENT_FEATURE_NAMES)


def precompute_news_sentiment(
    news_parquet_path: str,
    output_path: str,
    model_type: str = 'finbert',
    text_column: str = 'title',
    device: str = 'cuda'
):
    """
    Precompute sentiment for all news articles and save to parquet.
    This should be run once to avoid recomputing sentiment each time.
    
    Args:
        news_parquet_path: Path to news embeddings parquet
        output_path: Where to save the enhanced parquet
        model_type: 'finbert', 'vader', or 'textblob'
        text_column: Column with text to analyze
        device: 'cuda' or 'cpu'
    """
    print(f"\nPrecomputing sentiment for: {news_parquet_path}")
    
    # Load news
    df = pd.read_parquet(news_parquet_path)
    print(f"  Total articles: {len(df)}")
    
    # Check if sentiment already exists
    if 'sentiment_compound' in df.columns:
        print("  ✓ Sentiment already computed, skipping")
        return
    
    # Get analyzer
    analyzer = get_sentiment_analyzer(model_type, device)
    
    # Get texts
    texts = df[text_column].fillna('').tolist()
    
    # Analyze
    sentiments = analyzer.analyze_batch(texts, batch_size=64)
    
    # Add to dataframe
    df['sentiment_compound'] = [s['compound'] for s in sentiments]
    df['sentiment_positive'] = [s['positive'] for s in sentiments]
    df['sentiment_negative'] = [s['negative'] for s in sentiments]
    df['sentiment_neutral'] = [s['neutral'] for s in sentiments]
    
    # Save
    df.to_parquet(output_path)
    print(f"  ✓ Saved to: {output_path}")
    
    # Print stats
    compounds = df['sentiment_compound']
    print(f"  Sentiment stats:")
    print(f"    Mean: {compounds.mean():.3f}")
    print(f"    Std:  {compounds.std():.3f}")
    print(f"    Positive (>0.1): {(compounds > 0.1).mean()*100:.1f}%")
    print(f"    Negative (<-0.1): {(compounds < -0.1).mean()*100:.1f}%")


if __name__ == "__main__":
    # Test sentiment analyzers
    test_texts = [
        "Apple stock surges after strong earnings report beats expectations",
        "Tesla shares plummet amid concerns over declining demand",
        "Microsoft announces new partnership with OpenAI",
        "Market remains stable despite economic uncertainty",
        "Amazon faces regulatory challenges in Europe"
    ]
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS TEST")
    print("="*60)
    
    # Test VADER (always available)
    try:
        vader = VADERSentiment()
        print("\nVADER Results:")
        for text in test_texts:
            result = vader.analyze(text)
            print(f"  {result['compound']:+.3f} | {text[:50]}...")
    except ImportError as e:
        print(f"VADER not available: {e}")
    
    # Test FinBERT (requires transformers)
    try:
        finbert = FinBERTSentiment(device='cuda')
        print("\nFinBERT Results:")
        for text in test_texts:
            result = finbert.analyze(text)
            print(f"  {result['compound']:+.3f} | {text[:50]}...")
    except Exception as e:
        print(f"FinBERT not available: {e}")

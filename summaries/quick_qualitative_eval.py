"""
Quick qualitative evaluation - Shows 3 random articles with all model summaries
"""

import pandas as pd
import numpy as np
from pathlib import Path
from textwrap import fill
import random


def format_text(text, width=100):
    """Format text with word wrapping"""
    if pd.isna(text):
        return "N/A"
    return fill(str(text), width=width)


def print_sep(char='=', length=100):
    """Print separator"""
    print(char * length)


def main():
    print("\n" + "="*100)
    print("QUALITATIVE EVALUATION - SUMMARY GENERATION MODELS")
    print("="*100)
    
    # Load data
    inference_dir = Path(__file__).parent.parent / "data" / "news" / "inference"
    
    models = {
        'pointer_generator': 'apple_news_pointer_generator.parquet',
        'pointer_generator_fasttext': 'apple_news_pointer_generator_fasttext.parquet',
        'simple_seq2seq': 'apple_news_simple_seq2seq.parquet',
        'simple_seq2seq_fasttext': 'apple_news_simple_seq2seq_fasttext.parquet',
        'transformer': 'apple_news_transformer.parquet',
        'transformer_fasttext': 'apple_news_transformer_fasttext.parquet'
    }
    
    print("\nLoading model outputs...")
    dataframes = {}
    for model_name, filename in models.items():
        file_path = inference_dir / filename
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dataframes[model_name] = df
            print(f"  ✓ {model_name}: {len(df)} articles")
    
    ref_df = list(dataframes.values())[0]
    
    # Select 3 random articles
    random.seed(42)  # For reproducibility
    selected_indices = random.sample(range(len(ref_df)), 3)
    
    print(f"\n✓ Selected articles: {selected_indices}\n")
    
    # Model info
    model_info = {
        'pointer_generator_fasttext': ('🥇 Pointer-Generator + FastText', 1, 'RNN'),
        'transformer_fasttext': ('🥈 Transformer + FastText', 2, 'Transformer'),
        'simple_seq2seq_fasttext': ('🥉 Simple Seq2Seq + FastText', 3, 'RNN'),
        'pointer_generator': ('4️⃣ Pointer-Generator', 4, 'RNN'),
        'simple_seq2seq': ('5️⃣ Simple Seq2Seq', 5, 'RNN'),
        'transformer': ('6️⃣ Transformer', 6, 'Transformer')
    }
    
    # Process each article
    for i, idx in enumerate(selected_indices, 1):
        article = ref_df.iloc[idx]
        
        print_sep('=')
        print(f"ARTICLE #{i} (Index: {idx})")
        print_sep('=')
        
        print("\n📰 ORIGINAL TEXT:")
        print_sep('-')
        print(format_text(article['clean_body']))
        
        print("\n\n✅ REFERENCE SUMMARY:")
        print_sep('-')
        print(format_text(article['body_summary']))
        print(f"   [{len(article['body_summary'].split())} words]")
        
        print("\n\n🤖 GENERATED SUMMARIES:")
        print_sep('=')
        
        # Show summaries in rank order
        for model_key in ['pointer_generator_fasttext', 'transformer_fasttext',
                         'simple_seq2seq_fasttext', 'pointer_generator',
                         'simple_seq2seq', 'transformer']:
            if model_key in dataframes and idx < len(dataframes[model_key]):
                name, rank, model_type = model_info[model_key]
                summary = dataframes[model_key].iloc[idx]['generated_summary']
                word_count = len(str(summary).split())
                
                print(f"\n{name} [{model_type}]")
                print_sep('-')
                print(format_text(summary))
                print(f"   [{word_count} words]")
        
        print("\n\n")
    
    print_sep('=')
    print("EVALUATION COMPLETE")
    print_sep('=')
    print("\n💡 Observations to consider:")
    print("  • Faithfulness: Accuracy to original content")
    print("  • Fluency: Natural language flow")
    print("  • Coverage: Important facts included")
    print("  • Conciseness: Appropriate length")
    print("  • Factual precision: Names, numbers, dates correct")
    print("\n")


if __name__ == "__main__":
    main()

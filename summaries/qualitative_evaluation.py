"""
Qualitative evaluation script for summary generation models.
Displays side-by-side comparison of summaries from all models for selected news articles.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from textwrap import fill
import random


def format_text(text, width=80):
    """Format text with word wrapping"""
    if pd.isna(text):
        return "N/A"
    return fill(str(text), width=width)


def print_separator(char='=', length=100):
    """Print a separator line"""
    print(char * length)


def print_article_comparison(article_data, article_num, models_data):
    """
    Print a comprehensive comparison for a single article
    
    Args:
        article_data: Dictionary with article info from one model
        article_num: Article number for display
        models_data: Dictionary with data from all models
    """
    print_separator('=')
    print(f"ARTICLE #{article_num}")
    print_separator('=')
    
    # Display original text and reference summary
    print("\n📰 ORIGINAL TEXT:")
    print_separator('-')
    print(format_text(article_data['clean_body'], width=100))
    
    print("\n\n✅ REFERENCE SUMMARY (Ground Truth):")
    print_separator('-')
    print(format_text(article_data['body_summary'], width=100))
    
    print("\n\n🤖 GENERATED SUMMARIES BY MODEL:")
    print_separator('=')
    
    # Define model order and display names
    model_info = {
        'pointer_generator_fasttext': {
            'name': 'Pointer-Generator + FastText',
            'rank': 1,
            'type': 'RNN',
            'emoji': '🥇'
        },
        'transformer_fasttext': {
            'name': 'Transformer + FastText',
            'rank': 2,
            'type': 'Transformer',
            'emoji': '🥈'
        },
        'simple_seq2seq_fasttext': {
            'name': 'Simple Seq2Seq + FastText',
            'rank': 3,
            'type': 'RNN',
            'emoji': '🥉'
        },
        'pointer_generator': {
            'name': 'Pointer-Generator',
            'rank': 4,
            'type': 'RNN',
            'emoji': '4️⃣'
        },
        'simple_seq2seq': {
            'name': 'Simple Seq2Seq',
            'rank': 5,
            'type': 'RNN',
            'emoji': '5️⃣'
        },
        'transformer': {
            'name': 'Transformer',
            'rank': 6,
            'type': 'Transformer',
            'emoji': '6️⃣'
        }
    }
    
    # Sort models by rank
    sorted_models = sorted(model_info.items(), key=lambda x: x[1]['rank'])
    
    for model_key, info in sorted_models:
        if model_key in models_data:
            summary = models_data[model_key]
            print(f"\n{info['emoji']} {info['name']} [{info['type']}] - Rank #{info['rank']}")
            print_separator('-')
            print(format_text(summary, width=100))
    
    print("\n")


def calculate_summary_stats(summary):
    """Calculate basic statistics for a summary"""
    if pd.isna(summary):
        return {'words': 0, 'chars': 0}
    
    text = str(summary)
    return {
        'words': len(text.split()),
        'chars': len(text)
    }


def main():
    """Main function"""
    print("\n" + "="*100)
    print("QUALITATIVE EVALUATION OF SUMMARY GENERATION MODELS")
    print("="*100)
    
    # Define paths
    inference_dir = Path(__file__).parent.parent / "data" / "news" / "inference"
    
    # Load all model outputs
    print("\nLoading model outputs...")
    models = {
        'pointer_generator': 'apple_news_pointer_generator.parquet',
        'pointer_generator_fasttext': 'apple_news_pointer_generator_fasttext.parquet',
        'simple_seq2seq': 'apple_news_simple_seq2seq.parquet',
        'simple_seq2seq_fasttext': 'apple_news_simple_seq2seq_fasttext.parquet',
        'transformer': 'apple_news_transformer.parquet',
        'transformer_fasttext': 'apple_news_transformer_fasttext.parquet'
    }
    
    dataframes = {}
    for model_name, filename in models.items():
        file_path = inference_dir / filename
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dataframes[model_name] = df
            print(f"  ✓ Loaded {model_name}: {len(df)} articles")
        else:
            print(f"  ✗ Missing {model_name}: {file_path}")
    
    if not dataframes:
        print("\n❌ No model outputs found!")
        return
    
    # Get reference dataframe (use first available)
    ref_df = list(dataframes.values())[0]
    num_articles = len(ref_df)
    
    print(f"\n✓ Total articles available: {num_articles}")
    
    # Selection strategy
    print("\nSelect evaluation mode:")
    print("  1. Random selection (3-4 articles)")
    print("  2. Manual selection by indices")
    print("  3. Articles with longest original text")
    print("  4. Articles with shortest original text")
    print("  5. Diverse length selection")
    
    try:
        choice = input("\nEnter choice (1-5) [default: 1]: ").strip()
        if not choice:
            choice = '1'
        
        if choice == '1':
            # Random selection
            num_samples = int(input("Number of articles to sample (3-4 recommended) [default: 4]: ").strip() or "4")
            selected_indices = random.sample(range(num_articles), min(num_samples, num_articles))
            
        elif choice == '2':
            # Manual selection
            indices_input = input("Enter article indices separated by commas (e.g., 0,100,500,1000): ").strip()
            selected_indices = [int(idx.strip()) for idx in indices_input.split(',')]
            selected_indices = [idx for idx in selected_indices if 0 <= idx < num_articles]
            
        elif choice == '3':
            # Longest articles
            ref_df['text_length'] = ref_df['clean_body'].apply(lambda x: len(str(x).split()))
            selected_indices = ref_df.nlargest(4, 'text_length').index.tolist()
            
        elif choice == '4':
            # Shortest articles
            ref_df['text_length'] = ref_df['clean_body'].apply(lambda x: len(str(x).split()))
            selected_indices = ref_df.nsmallest(4, 'text_length').index.tolist()
            
        elif choice == '5':
            # Diverse length selection
            ref_df['text_length'] = ref_df['clean_body'].apply(lambda x: len(str(x).split()))
            quantiles = [0.1, 0.35, 0.65, 0.9]
            selected_indices = []
            for q in quantiles:
                quantile_val = ref_df['text_length'].quantile(q)
                idx = (ref_df['text_length'] - quantile_val).abs().idxmin()
                selected_indices.append(idx)
        else:
            print("Invalid choice, using random selection")
            selected_indices = random.sample(range(num_articles), 4)
    
    except (ValueError, KeyboardInterrupt):
        print("\nUsing default: random selection of 4 articles")
        selected_indices = random.sample(range(num_articles), 4)
    
    print(f"\n✓ Selected {len(selected_indices)} articles for evaluation")
    print(f"  Indices: {selected_indices}")
    
    # Process each selected article
    for i, idx in enumerate(selected_indices, 1):
        # Get article data from first available dataframe
        article_data = ref_df.iloc[idx]
        
        # Collect summaries from all models
        models_data = {}
        for model_name, df in dataframes.items():
            if idx < len(df):
                models_data[model_name] = df.iloc[idx]['generated_summary']
        
        # Print comparison
        print_article_comparison(article_data, i, models_data)
        
        # Show statistics
        print("📊 SUMMARY STATISTICS:")
        print_separator('-')
        
        ref_stats = calculate_summary_stats(article_data['body_summary'])
        print(f"Reference:     {ref_stats['words']:3d} words, {ref_stats['chars']:4d} chars")
        
        for model_name in ['pointer_generator_fasttext', 'transformer_fasttext', 
                          'simple_seq2seq_fasttext', 'pointer_generator', 
                          'simple_seq2seq', 'transformer']:
            if model_name in models_data:
                stats = calculate_summary_stats(models_data[model_name])
                model_display = model_name.replace('_', ' ').title()
                print(f"{model_display:25s} {stats['words']:3d} words, {stats['chars']:4d} chars")
        
        print("\n")
        
        # Pause between articles (except for the last one)
        if i < len(selected_indices):
            input("Press Enter to continue to next article...")
            print("\n\n")
    
    # Final summary
    print_separator('=')
    print("EVALUATION COMPLETE")
    print_separator('=')
    print("\n💡 Key observations to consider:")
    print("  • Faithfulness: Does the summary accurately represent the original?")
    print("  • Conciseness: Is the summary appropriately condensed?")
    print("  • Coherence: Does the summary read naturally?")
    print("  • Completeness: Are key facts preserved?")
    print("  • Specific facts: Are names, numbers, dates correctly copied?")
    print("\n")
    
    # Option to save results
    save_output = input("Would you like to save this comparison to a file? (y/n) [default: n]: ").strip().lower()
    
    if save_output == 'y':
        output_dir = Path(__file__).parent / "qualitative_results"
        output_dir.mkdir(exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"qualitative_eval_{timestamp}.txt"
        
        # Save to file (redirect output)
        import sys
        original_stdout = sys.stdout
        
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            for i, idx in enumerate(selected_indices, 1):
                article_data = ref_df.iloc[idx]
                models_data = {}
                for model_name, df in dataframes.items():
                    if idx < len(df):
                        models_data[model_name] = df.iloc[idx]['generated_summary']
                
                print_article_comparison(article_data, i, models_data)
                
                # Statistics
                print("📊 SUMMARY STATISTICS:")
                print_separator('-')
                ref_stats = calculate_summary_stats(article_data['body_summary'])
                print(f"Reference:     {ref_stats['words']:3d} words, {ref_stats['chars']:4d} chars")
                
                for model_name in ['pointer_generator_fasttext', 'transformer_fasttext', 
                                  'simple_seq2seq_fasttext', 'pointer_generator', 
                                  'simple_seq2seq', 'transformer']:
                    if model_name in models_data:
                        stats = calculate_summary_stats(models_data[model_name])
                        model_display = model_name.replace('_', ' ').title()
                        print(f"{model_display:25s} {stats['words']:3d} words, {stats['chars']:4d} chars")
                print("\n\n")
        
        sys.stdout = original_stdout
        print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

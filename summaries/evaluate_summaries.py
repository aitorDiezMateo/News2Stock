"""
Evaluate generated summaries using ROUGE and BLEU metrics
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import evaluation metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    print("Installing required packages...")
    os.system("pip install rouge-score nltk -q")
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)


def tokenize_for_bleu(text):
    """Tokenize text for BLEU score calculation"""
    if pd.isna(text) or text == '':
        return []
    return str(text).lower().split()


def calculate_rouge(reference, generated):
    """Calculate ROUGE scores"""
    if pd.isna(reference) or pd.isna(generated):
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(str(reference), str(generated))
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def calculate_bleu(reference, generated):
    """Calculate BLEU score with smoothing"""
    reference_tokens = tokenize_for_bleu(reference)
    generated_tokens = tokenize_for_bleu(generated)
    
    if len(reference_tokens) == 0 or len(generated_tokens) == 0:
        return 0.0
    
    # Use smoothing function to handle cases with no n-gram matches
    smoothing = SmoothingFunction().method1
    
    try:
        # Calculate BLEU score (reference should be in a list of lists)
        score = sentence_bleu([reference_tokens], generated_tokens, 
                            smoothing_function=smoothing)
        return score
    except:
        return 0.0


def calculate_summary_length_ratio(reference, generated):
    """Calculate the ratio of generated summary length to reference length"""
    ref_len = len(str(reference).split())
    gen_len = len(str(generated).split())
    
    if ref_len == 0:
        return 0.0
    
    return gen_len / ref_len


def evaluate_model(df, model_name):
    """Evaluate a single model's summaries"""
    print(f"\nEvaluating {model_name}...")
    print(f"Number of summaries: {len(df)}")
    
    # Calculate metrics for each summary
    rouge_scores = []
    bleu_scores = []
    length_ratios = []
    
    for idx, row in df.iterrows():
        reference = row['body_summary']
        generated = row['generated_summary']
        
        # ROUGE scores
        rouge = calculate_rouge(reference, generated)
        rouge_scores.append(rouge)
        
        # BLEU score
        bleu = calculate_bleu(reference, generated)
        bleu_scores.append(bleu)
        
        # Length ratio
        length_ratio = calculate_summary_length_ratio(reference, generated)
        length_ratios.append(length_ratio)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} summaries...")
    
    # Aggregate results
    results = {
        'model': model_name,
        'num_summaries': len(df),
        'rouge1_mean': np.mean([s['rouge1'] for s in rouge_scores]),
        'rouge1_std': np.std([s['rouge1'] for s in rouge_scores]),
        'rouge2_mean': np.mean([s['rouge2'] for s in rouge_scores]),
        'rouge2_std': np.std([s['rouge2'] for s in rouge_scores]),
        'rougeL_mean': np.mean([s['rougeL'] for s in rouge_scores]),
        'rougeL_std': np.std([s['rougeL'] for s in rouge_scores]),
        'bleu_mean': np.mean(bleu_scores),
        'bleu_std': np.std(bleu_scores),
        'length_ratio_mean': np.mean(length_ratios),
        'length_ratio_std': np.std(length_ratios)
    }
    
    return results


def print_results(results):
    """Print results in a formatted table"""
    print("\n" + "="*100)
    print("EVALUATION RESULTS - Summary Generation Metrics")
    print("="*100)
    print(f"\n{'Model':<30} {'Samples':<10} {'ROUGE-1':<15} {'ROUGE-2':<15} {'ROUGE-L':<15} {'BLEU':<12} {'Len Ratio':<12}")
    print("-"*100)
    
    for result in results:
        model_name = result['model'].replace('apple_news_', '')
        print(f"{model_name:<30} {result['num_summaries']:<10} "
              f"{result['rouge1_mean']:.4f}±{result['rouge1_std']:.4f}  "
              f"{result['rouge2_mean']:.4f}±{result['rouge2_std']:.4f}  "
              f"{result['rougeL_mean']:.4f}±{result['rougeL_std']:.4f}  "
              f"{result['bleu_mean']:.4f}±{result['bleu_std']:.4f}  "
              f"{result['length_ratio_mean']:.2f}±{result['length_ratio_std']:.2f}")
    
    print("="*100)
    print("\nMetric Explanations:")
    print("  ROUGE-1: Overlap of unigrams (single words) between reference and generated")
    print("  ROUGE-2: Overlap of bigrams (two consecutive words)")
    print("  ROUGE-L: Longest common subsequence between reference and generated")
    print("  BLEU: Precision-based metric considering n-grams up to 4-grams")
    print("  Len Ratio: Generated summary length / Reference summary length (1.0 = same length)")
    print("\nNote: Higher scores are better for ROUGE and BLEU (range 0-1)")
    print("      Len Ratio close to 1.0 indicates similar length to reference")
    print("="*100)


def save_detailed_results(results, output_path):
    """Save detailed results to CSV"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to: {output_path}")


def analyze_per_sample(inference_dir, output_dir):
    """Calculate metrics per sample and save individual scores"""
    print("\n" + "="*100)
    print("CALCULATING PER-SAMPLE METRICS")
    print("="*100)
    
    models = [
        'transformer',
        'transformer_fasttext',
        'simple_seq2seq',
        'simple_seq2seq_fasttext',
        'pointer_generator',
        'pointer_generator_fasttext'
    ]
    
    for model in models:
        file_path = os.path.join(inference_dir, f'apple_news_{model}.parquet')
        
        if not os.path.exists(file_path):
            print(f"  ✗ Skipping {model}: file not found")
            continue
        
        print(f"\nProcessing {model}...")
        df = pd.read_parquet(file_path)
        
        # Calculate metrics for each sample
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        
        for idx, row in df.iterrows():
            rouge = calculate_rouge(row['body_summary'], row['generated_summary'])
            bleu = calculate_bleu(row['body_summary'], row['generated_summary'])
            
            rouge1_scores.append(rouge['rouge1'])
            rouge2_scores.append(rouge['rouge2'])
            rougeL_scores.append(rouge['rougeL'])
            bleu_scores.append(bleu)
            
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples...")
        
        # Add scores to dataframe
        df['rouge1'] = rouge1_scores
        df['rouge2'] = rouge2_scores
        df['rougeL'] = rougeL_scores
        df['bleu'] = bleu_scores
        
        # Save with scores
        output_path = os.path.join(output_dir, f'apple_news_{model}_with_scores.parquet')
        df.to_parquet(output_path, index=False)
        print(f"  ✓ Saved with scores to: {output_path}")


def main(save_per_sample=False):
    """Main evaluation pipeline
    
    Args:
        save_per_sample: If True, save individual scores for each sample
    """
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to News2Stock
    INFERENCE_DIR = os.path.join(ROOT_DIR, 'data', 'news', 'inference')
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')  # results in summaries folder
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*100)
    print("SUMMARIZATION EVALUATION PIPELINE")
    print("="*100)
    print(f"Inference directory: {INFERENCE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Models to evaluate
    models = [
        "transformer",
        "transformer_fasttext",
        'simple_seq2seq',
        'simple_seq2seq_fasttext',
        'pointer_generator',
        'pointer_generator_fasttext'
        
    ]
    
    # Evaluate each model
    all_results = []
    
    for model in models:
        file_path = os.path.join(INFERENCE_DIR, f'apple_news_{model}.parquet')
        
        if not os.path.exists(file_path):
            print(f"\n✗ Skipping {model}: file not found at {file_path}")
            continue
        
        # Load data
        df = pd.read_parquet(file_path)
        
        # Evaluate
        results = evaluate_model(df, f'apple_news_{model}')
        all_results.append(results)
    
    # Print comparative results
    if all_results:
        print_results(all_results)
        
        # Save summary results
        summary_path = os.path.join(RESULTS_DIR, 'evaluation_summary.csv')
        save_detailed_results(all_results, summary_path)
        
        # Optionally calculate and save per-sample scores
        if save_per_sample:
            analyze_per_sample(INFERENCE_DIR, RESULTS_DIR)
    else:
        print("\n✗ No models were evaluated. Please check that inference files exist.")


if __name__ == "__main__":
    import sys
    
    # Check if --per-sample flag is provided
    save_per_sample = '--per-sample' in sys.argv
    
    if save_per_sample:
        print("\n*** Per-sample scores will be saved ***\n")
    
    main(save_per_sample=save_per_sample)


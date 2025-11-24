"""
Visualize evaluation results for summary generation models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def create_comparison_plot(df, output_dir):
    """Create a comprehensive comparison plot"""
    
    # Simplify model names
    df['model_short'] = df['model'].str.replace('apple_news_', '')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Summary Generation Models - Performance Comparison', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = sns.color_palette("husl", len(df))
    
    # 1. ROUGE Scores
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.25
    
    ax1.bar(x - width, df['rouge1_mean'], width, label='ROUGE-1', 
            yerr=df['rouge1_std'], capsize=5, color=colors[0], alpha=0.8)
    ax1.bar(x, df['rouge2_mean'], width, label='ROUGE-2', 
            yerr=df['rouge2_std'], capsize=5, color=colors[1], alpha=0.8)
    ax1.bar(x + width, df['rougeL_mean'], width, label='ROUGE-L', 
            yerr=df['rougeL_std'], capsize=5, color=colors[2], alpha=0.8)
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('ROUGE Scores (Higher is Better)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model_short'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. BLEU Scores
    ax2 = axes[0, 1]
    bars = ax2.bar(df['model_short'], df['bleu_mean'], 
                   yerr=df['bleu_std'], capsize=5, color=colors, alpha=0.8)
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('BLEU Score', fontweight='bold')
    ax2.set_title('BLEU Scores (Higher is Better)', fontweight='bold')
    ax2.set_xticklabels(df['model_short'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Length Ratio
    ax3 = axes[1, 0]
    bars = ax3.bar(df['model_short'], df['length_ratio_mean'], 
                   yerr=df['length_ratio_std'], capsize=5, color=colors, alpha=0.8)
    ax3.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Perfect ratio (1.0)')
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Length Ratio', fontweight='bold')
    ax3.set_title('Summary Length Ratio (Closer to 1.0 is Better)', fontweight='bold')
    ax3.set_xticklabels(df['model_short'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Overall Comparison (Radar-like)
    ax4 = axes[1, 1]
    
    # Normalize metrics to 0-1 scale for comparison
    metrics = ['rouge1_mean', 'rouge2_mean', 'rougeL_mean', 'bleu_mean']
    normalized_data = df[metrics].copy()
    
    # For length ratio, normalize by distance from 1.0 (inverted)
    length_ratio_normalized = 1 - np.abs(df['length_ratio_mean'] - 1.0)
    normalized_data['length_ratio'] = length_ratio_normalized
    
    # Plot grouped bar chart
    x = np.arange(len(df))
    width = 0.15
    
    for i, metric in enumerate(['rouge1_mean', 'rouge2_mean', 'rougeL_mean', 'bleu_mean']):
        offset = (i - 2) * width
        ax4.bar(x + offset, df[metric], width, 
                label=metric.replace('_mean', '').upper(), alpha=0.8)
    
    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.set_title('All Metrics Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['model_short'], rotation=45, ha='right')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    
    plt.close()


def create_ranking_table(df, output_dir):
    """Create a ranking table for each metric"""
    
    metrics = {
        'ROUGE-1': 'rouge1_mean',
        'ROUGE-2': 'rouge2_mean',
        'ROUGE-L': 'rougeL_mean',
        'BLEU': 'bleu_mean'
    }
    
    print("\n" + "="*80)
    print("MODEL RANKING BY METRIC")
    print("="*80)
    
    rankings = {}
    
    for metric_name, metric_col in metrics.items():
        df_sorted = df.sort_values(metric_col, ascending=False).copy()
        df_sorted['rank'] = range(1, len(df_sorted) + 1)
        rankings[metric_name] = df_sorted[['model', metric_col, 'rank']].copy()
        
        print(f"\n{metric_name} (Higher is Better):")
        print("-" * 80)
        for _, row in df_sorted.iterrows():
            model_name = row['model'].replace('apple_news_', '')
            print(f"  {row['rank']}. {model_name:<40} {row[metric_col]:.4f}")
    
    # Overall ranking (average rank across all metrics)
    print("\n" + "="*80)
    print("OVERALL RANKING (Average Rank Across All Metrics)")
    print("="*80)
    
    overall_ranks = pd.DataFrame({'model': df['model']})
    for metric_name, ranking_df in rankings.items():
        overall_ranks = overall_ranks.merge(
            ranking_df[['model', 'rank']].rename(columns={'rank': f'rank_{metric_name}'}),
            on='model'
        )
    
    # Calculate average rank
    rank_cols = [col for col in overall_ranks.columns if col.startswith('rank_')]
    overall_ranks['avg_rank'] = overall_ranks[rank_cols].mean(axis=1)
    overall_ranks = overall_ranks.sort_values('avg_rank')
    overall_ranks['final_rank'] = range(1, len(overall_ranks) + 1)
    
    for _, row in overall_ranks.iterrows():
        model_name = row['model'].replace('apple_news_', '')
        print(f"  {row['final_rank']}. {model_name:<40} (Avg Rank: {row['avg_rank']:.2f})")
    
    print("="*80)
    
    # Save ranking to CSV
    output_path = os.path.join(output_dir, 'model_rankings.csv')
    overall_ranks.to_csv(output_path, index=False)
    print(f"\n✓ Rankings saved to: {output_path}")


def create_detailed_comparison_table(df, output_dir):
    """Create a detailed LaTeX-style table"""
    
    print("\n" + "="*100)
    print("DETAILED COMPARISON TABLE (LaTeX Format)")
    print("="*100)
    
    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Summary Generation Models - Performance Comparison}")
    print("\\label{tab:model_comparison}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{ROUGE-1} & \\textbf{ROUGE-2} & \\textbf{ROUGE-L} & \\textbf{BLEU} & \\textbf{Length Ratio} \\\\")
    print("\\hline")
    
    for _, row in df.iterrows():
        model_name = row['model'].replace('apple_news_', '').replace('_', '\\_')
        print(f"{model_name} & "
              f"{row['rouge1_mean']:.4f} $\\pm$ {row['rouge1_std']:.4f} & "
              f"{row['rouge2_mean']:.4f} $\\pm$ {row['rouge2_std']:.4f} & "
              f"{row['rougeL_mean']:.4f} $\\pm$ {row['rougeL_std']:.4f} & "
              f"{row['bleu_mean']:.4f} $\\pm$ {row['bleu_std']:.4f} & "
              f"{row['length_ratio_mean']:.2f} $\\pm$ {row['length_ratio_std']:.2f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("="*100)


def analyze_fasttext_impact(df):
    """Analyze the impact of FastText embeddings"""
    
    print("\n" + "="*80)
    print("FASTTEXT EMBEDDINGS IMPACT ANALYSIS")
    print("="*80)
    
    # Simple Seq2Seq comparison
    simple_base = df[df['model'] == 'apple_news_simple_seq2seq'].iloc[0]
    simple_ft = df[df['model'] == 'apple_news_simple_seq2seq_fasttext'].iloc[0]
    
    print("\nSimple Seq2Seq: Base vs FastText")
    print("-" * 80)
    print(f"ROUGE-1:      {simple_base['rouge1_mean']:.4f} → {simple_ft['rouge1_mean']:.4f} "
          f"({((simple_ft['rouge1_mean']/simple_base['rouge1_mean'] - 1) * 100):+.2f}%)")
    print(f"ROUGE-2:      {simple_base['rouge2_mean']:.4f} → {simple_ft['rouge2_mean']:.4f} "
          f"({((simple_ft['rouge2_mean']/simple_base['rouge2_mean'] - 1) * 100):+.2f}%)")
    print(f"ROUGE-L:      {simple_base['rougeL_mean']:.4f} → {simple_ft['rougeL_mean']:.4f} "
          f"({((simple_ft['rougeL_mean']/simple_base['rougeL_mean'] - 1) * 100):+.2f}%)")
    print(f"BLEU:         {simple_base['bleu_mean']:.4f} → {simple_ft['bleu_mean']:.4f} "
          f"({((simple_ft['bleu_mean']/simple_base['bleu_mean'] - 1) * 100):+.2f}%)")
    
    # Pointer-Generator comparison
    pg_base = df[df['model'] == 'apple_news_pointer_generator'].iloc[0]
    pg_ft = df[df['model'] == 'apple_news_pointer_generator_fasttext'].iloc[0]
    
    print("\nPointer-Generator: Base vs FastText")
    print("-" * 80)
    print(f"ROUGE-1:      {pg_base['rouge1_mean']:.4f} → {pg_ft['rouge1_mean']:.4f} "
          f"({((pg_ft['rouge1_mean']/pg_base['rouge1_mean'] - 1) * 100):+.2f}%)")
    print(f"ROUGE-2:      {pg_base['rouge2_mean']:.4f} → {pg_ft['rouge2_mean']:.4f} "
          f"({((pg_ft['rouge2_mean']/pg_base['rouge2_mean'] - 1) * 100):+.2f}%)")
    print(f"ROUGE-L:      {pg_base['rougeL_mean']:.4f} → {pg_ft['rougeL_mean']:.4f} "
          f"({((pg_ft['rougeL_mean']/pg_base['rougeL_mean'] - 1) * 100):+.2f}%)")
    print(f"BLEU:         {pg_base['bleu_mean']:.4f} → {pg_ft['bleu_mean']:.4f} "
          f"({((pg_ft['bleu_mean']/pg_base['bleu_mean'] - 1) * 100):+.2f}%)")
    
    print("="*80)


def main():
    """Main visualization pipeline"""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    PLOTS_DIR = os.path.join(ROOT_DIR, 'plots', 'evaluation')
    
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load results
    results_path = os.path.join(RESULTS_DIR, 'evaluation_summary.csv')
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Please run evaluate_summaries.py first.")
        return
    
    df = pd.read_csv(results_path)
    
    print("="*80)
    print("VISUALIZATION PIPELINE")
    print("="*80)
    print(f"Loading results from: {results_path}")
    print(f"Plots will be saved to: {PLOTS_DIR}")
    
    # Create visualizations
    create_comparison_plot(df, PLOTS_DIR)
    create_ranking_table(df, RESULTS_DIR)
    create_detailed_comparison_table(df, RESULTS_DIR)
    analyze_fasttext_impact(df)
    
    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()


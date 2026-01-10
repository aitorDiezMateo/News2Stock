"""
Comparative Visualizations for Model Performance
=================================================
Creates bar charts comparing F1-score and Accuracy across:
- Different model architectures (Feedforward, LSTM, Attention)
- Different embedding types (patchtst, chronos, lstm_multihead)
- Different window sizes (5, 20)
- With/without news embeddings
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Paths
RESULTS_BASE = Path(__file__).parent.parent.parent / 'results'
PLOTS_PATH = Path(__file__).parent.parent.parent / 'plots' / 'prediction'
PLOTS_PATH.mkdir(parents=True, exist_ok=True)


def load_metrics(model_type, embedding_type, window_size, with_news=True):
    """
    Load metrics from CSV file.
    
    Args:
        model_type: 'prediction', 'prediction_lstm', or 'prediction_attention'
        embedding_type: 'patchtst', 'chronos', or 'lstm_multihead'
        window_size: 5 or 20
        with_news: True to load models with news, False for without news
    """
    results_path = RESULTS_BASE / model_type
    
    prefix = "" if with_news else "no_news_"
    filename = f"test_metrics_{prefix}{embedding_type}_{window_size}.csv"
    filepath = results_path / filename
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath)
    
    # Handle different CSV formats
    result = {
        'accuracy': df['accuracy'].values[0],
        'f1_macro': df['f1_macro'].values[0],
        'precision_macro': df['precision_macro'].values[0],
        'recall_macro': df['recall_macro'].values[0]
    }
    
    # f1_weighted is optional (not present in prediction_attention)
    if 'f1_weighted' in df.columns:
        result['f1_weighted'] = df['f1_weighted'].values[0]
    else:
        result['f1_weighted'] = df['f1_macro'].values[0]  # Use f1_macro as fallback
    
    return result


def collect_all_results():
    """Collect all available results from all model types."""
    model_types = ['prediction', 'prediction_lstm', 'prediction_attention']
    embedding_types = ['patchtst', 'chronos', 'lstm_multihead']
    window_sizes = [5, 20]
    
    model_labels = {
        'prediction': 'Feedforward',
        'prediction_lstm': 'LSTM',
        'prediction_attention': 'Attention'
    }
    
    results = []
    
    for model_type in model_types:
        for emb_type in embedding_types:
            for window in window_sizes:
                # With news
                metrics_with = load_metrics(model_type, emb_type, window, with_news=True)
                if metrics_with:
                    results.append({
                        'model_type': model_type,
                        'model_label': model_labels[model_type],
                        'embedding': emb_type,
                        'window': window,
                        'has_news': 'With News',
                        **metrics_with
                    })
                
                # Without news
                metrics_without = load_metrics(model_type, emb_type, window, with_news=False)
                if metrics_without:
                    results.append({
                        'model_type': model_type,
                        'model_label': model_labels[model_type],
                        'embedding': emb_type,
                        'window': window,
                        'has_news': 'Without News',
                        **metrics_without
                    })
    
    return pd.DataFrame(results)


def plot_comparison_by_embedding_and_window(df, metric='f1_macro', title_suffix=''):
    """
    Create 6 subplots: one for each (embedding_type, window_size) combination.
    Each subplot compares all model types (Feedforward, LSTM, Attention) with/without news.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'Model Architecture Comparison: {title_suffix}', 
                fontsize=16, fontweight='bold', y=0.998)
    
    embedding_types = ['patchtst', 'chronos', 'lstm_multihead']
    window_sizes = [5, 20]
    
    embedding_labels = {
        'patchtst': 'PatchTST',
        'chronos': 'Chronos',
        'lstm_multihead': 'LSTM-Multihead'
    }
    
    # Colors for different model types
    colors_with = {'Feedforward': '#3498db', 'LSTM': '#2ecc71', 'Attention': '#9b59b6'}
    colors_without = {'Feedforward': '#e74c3c', 'LSTM': '#e67e22', 'Attention': '#95a5a6'}
    
    for i, window in enumerate(window_sizes):
        for j, emb_type in enumerate(embedding_types):
            ax = axes[i, j]
            
            # Filter data
            subset = df[(df['embedding'] == emb_type) & (df['window'] == window)].copy()
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{embedding_labels[emb_type]} (W={window})')
                continue
            
            # Sort by model_label and has_news for consistent ordering
            subset = subset.sort_values(['model_label', 'has_news'], 
                                       ascending=[True, False])
            
            # Create grouped bar chart
            n_items = len(subset)
            x_pos = np.arange(n_items)
            
            # Assign colors based on model and news status
            bar_colors = []
            for _, row in subset.iterrows():
                if row['has_news'] == 'With News':
                    bar_colors.append(colors_with[row['model_label']])
                else:
                    bar_colors.append(colors_without[row['model_label']])
            
            bars = ax.bar(x_pos, subset[metric].values, 
                         color=bar_colors, alpha=0.85, 
                         edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar, val in zip(bars, subset[metric].values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Create labels combining model type and news status
            labels = [f"{row['model_label']}\n{'w/ News' if row['has_news']=='With News' else 'No News'}" 
                     for _, row in subset.iterrows()]
            
            # Styling
            ax.set_ylabel(title_suffix, fontsize=10, fontweight='bold')
            ax.set_title(f'{embedding_labels[emb_type]} (Window={window})', 
                        fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=0, fontsize=8)
            ax.set_ylim([0, max(0.7, subset[metric].max() * 1.15)])
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_embedding_comparison_by_window(df, metric='f1_macro', title_suffix=''):
    """
    Create 2 subplots: one for each window size.
    Each subplot compares all embedding types across all model architectures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Embedding Type Comparison Across Architectures: {title_suffix}', 
                fontsize=16, fontweight='bold', y=1.02)
    
    window_sizes = [5, 20]
    embedding_types = ['patchtst', 'chronos', 'lstm_multihead']
    model_labels = ['Feedforward', 'LSTM', 'Attention']
    
    embedding_labels = {
        'patchtst': 'PatchTST',
        'chronos': 'Chronos',
        'lstm_multihead': 'LSTM-MH'
    }
    
    colors = {'Feedforward': '#3498db', 'LSTM': '#2ecc71', 'Attention': '#9b59b6'}
    
    for idx, window in enumerate(window_sizes):
        ax = axes[idx]
        
        # Filter data - only with news for cleaner comparison
        subset = df[(df['window'] == window) & (df['has_news'] == 'With News')].copy()
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Window Size = {window}')
            continue
        
        # Prepare grouped bar chart data
        x_labels = [embedding_labels[e] for e in embedding_types]
        x_pos = np.arange(len(embedding_types))
        width = 0.25
        
        # Get values for each model type
        model_data = {}
        for model in model_labels:
            model_data[model] = []
            for emb in embedding_types:
                data = subset[(subset['embedding'] == emb) & (subset['model_label'] == model)]
                val = data[metric].values[0] if len(data) > 0 else 0
                model_data[model].append(val)
        
        # Create grouped bars
        for i, model in enumerate(model_labels):
            offset = (i - 1) * width
            bars = ax.bar(x_pos + offset, model_data[model], width,
                         label=model, color=colors[model], alpha=0.85,
                         edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bar, val in zip(bars, model_data[model]):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{val:.3f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Stock Embedding Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(title_suffix, fontsize=12, fontweight='bold')
        ax.set_title(f'Window Size = {window} days', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend(title='Architecture', loc='upper right', fontsize=10)
        
        # Get max value for y-limit
        all_vals = [v for vals in model_data.values() for v in vals if v > 0]
        ax.set_ylim([0, max(0.7, max(all_vals) * 1.15) if all_vals else 0.7])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_window_comparison_by_embedding(df, metric='f1_macro', title_suffix=''):
    """
    Create 3 subplots: one for each embedding type.
    Each subplot compares window sizes across all model architectures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Window Size Comparison Across Architectures: {title_suffix}', 
                fontsize=16, fontweight='bold', y=1.02)
    
    embedding_types = ['patchtst', 'chronos', 'lstm_multihead']
    window_sizes = [5, 20]
    model_labels = ['Feedforward', 'LSTM', 'Attention']
    
    embedding_labels = {
        'patchtst': 'PatchTST',
        'chronos': 'Chronos',
        'lstm_multihead': 'LSTM-Multihead'
    }
    
    colors = {'Feedforward': '#3498db', 'LSTM': '#2ecc71', 'Attention': '#9b59b6'}
    
    for idx, emb_type in enumerate(embedding_types):
        ax = axes[idx]
        
        # Filter data - only with news for cleaner comparison
        subset = df[(df['embedding'] == emb_type) & (df['has_news'] == 'With News')].copy()
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(embedding_labels[emb_type])
            continue
        
        # Prepare data for grouped bar chart
        x_labels = [f'W={w}' for w in window_sizes]
        x_pos = np.arange(len(window_sizes))
        width = 0.25
        
        # Get values for each model type
        model_data = {}
        for model in model_labels:
            model_data[model] = []
            for window in window_sizes:
                data = subset[(subset['window'] == window) & (subset['model_label'] == model)]
                val = data[metric].values[0] if len(data) > 0 else 0
                model_data[model].append(val)
        
        # Create grouped bars
        for i, model in enumerate(model_labels):
            offset = (i - 1) * width
            bars = ax.bar(x_pos + offset, model_data[model], width,
                         label=model, color=colors[model], alpha=0.85,
                         edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bar, val in zip(bars, model_data[model]):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{val:.3f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(title_suffix, fontsize=12, fontweight='bold')
        ax.set_title(embedding_labels[emb_type], fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend(title='Architecture', loc='upper right', fontsize=10)
        
        # Get max value for y-limit
        all_vals = [v for vals in model_data.values() for v in vals if v > 0]
        ax.set_ylim([0, max(0.7, max(all_vals) * 1.15) if all_vals else 0.7])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def create_summary_table(df):
    """Create a summary table with all results."""
    # Pivot table
    summary = df.pivot_table(
        values=['accuracy', 'f1_macro', 'f1_weighted'],
        index=['model_label', 'embedding', 'window'],
        columns='has_news',
        aggfunc='first'
    )
    
    # Save to CSV
    output_path = PLOTS_PATH / 'model_comparison_summary.csv'
    summary.to_csv(output_path)
    print(f"✅ Summary table saved to: {output_path}")
    
    # Also create a simplified comparison table
    simple_summary = df.groupby(['model_label', 'embedding', 'window', 'has_news'])[
        ['accuracy', 'f1_macro']
    ].first().reset_index()
    
    simple_path = PLOTS_PATH / 'model_comparison_simple.csv'
    simple_summary.to_csv(simple_path, index=False)
    print(f"✅ Simple summary saved to: {simple_path}")
    
    return summary


def main():
    """Generate all comparison visualizations."""
    print("\n" + "=" * 70)
    print("GENERATING MODEL COMPARISON VISUALIZATIONS")
    print("=" * 70)
    
    # Collect results
    print("\n📊 Collecting results...")
    df = collect_all_results()
    
    if len(df) == 0:
        print("❌ No results found!")
        return
    
    print(f"✅ Loaded {len(df)} result entries")
    print(f"   Model Architectures: {df['model_label'].unique()}")
    print(f"   Embeddings: {df['embedding'].unique()}")
    print(f"   Windows: {df['window'].unique()}")
    print(f"   Has News: {df['has_news'].unique()}")
    
    # Create summary table
    print("\n📋 Creating summary table...")
    summary = create_summary_table(df)
    print("\nSummary Statistics (first 10 rows):")
    print(summary.head(10).round(4))
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...\n")
    
    # 1. Individual comparison plots (6 subplots) - F1 Score
    print("1. Creating detailed comparison by embedding and window (F1-Macro)...")
    fig1 = plot_comparison_by_embedding_and_window(df, metric='f1_macro', 
                                                   title_suffix='F1-Score (Macro)')
    fig1.savefig(PLOTS_PATH / 'comparison_detailed_f1_macro.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"   ✅ Saved: comparison_detailed_f1_macro.png")
    
    # 2. Individual comparison plots (6 subplots) - Accuracy
    print("2. Creating detailed comparison by embedding and window (Accuracy)...")
    fig2 = plot_comparison_by_embedding_and_window(df, metric='accuracy',
                                                   title_suffix='Accuracy')
    fig2.savefig(PLOTS_PATH / 'comparison_detailed_accuracy.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"   ✅ Saved: comparison_detailed_accuracy.png")
    
    # 3. Embedding comparison by window - F1 Score
    print("3. Creating embedding type comparison (F1-Macro)...")
    fig3 = plot_embedding_comparison_by_window(df, metric='f1_macro',
                                              title_suffix='F1-Score (Macro)')
    fig3.savefig(PLOTS_PATH / 'comparison_embeddings_f1_macro.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"   ✅ Saved: comparison_embeddings_f1_macro.png")
    
    # 4. Embedding comparison by window - Accuracy
    print("4. Creating embedding type comparison (Accuracy)...")
    fig4 = plot_embedding_comparison_by_window(df, metric='accuracy',
                                              title_suffix='Accuracy')
    fig4.savefig(PLOTS_PATH / 'comparison_embeddings_accuracy.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"   ✅ Saved: comparison_embeddings_accuracy.png")
    
    # 5. Window size comparison by embedding - F1 Score
    print("5. Creating window size comparison (F1-Macro)...")
    fig5 = plot_window_comparison_by_embedding(df, metric='f1_macro',
                                              title_suffix='F1-Score (Macro)')
    fig5.savefig(PLOTS_PATH / 'comparison_windows_f1_macro.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print(f"   ✅ Saved: comparison_windows_f1_macro.png")
    
    # 6. Window size comparison by embedding - Accuracy
    print("6. Creating window size comparison (Accuracy)...")
    fig6 = plot_window_comparison_by_embedding(df, metric='accuracy',
                                              title_suffix='Accuracy')
    fig6.savefig(PLOTS_PATH / 'comparison_windows_accuracy.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig6)
    print(f"   ✅ Saved: comparison_windows_accuracy.png")
    
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {PLOTS_PATH}")
    print("=" * 70)
    
    # Show best performing models
    print("\n🏆 BEST PERFORMING MODELS:")
    print("\nBy F1-Macro Score:")
    best_f1 = df.loc[df['f1_macro'].idxmax()]
    print(f"  Architecture: {best_f1['model_label']}")
    print(f"  Embedding: {best_f1['embedding']} (Window={best_f1['window']})")
    print(f"  News: {best_f1['has_news']}")
    print(f"  F1-Macro: {best_f1['f1_macro']:.4f}, Accuracy: {best_f1['accuracy']:.4f}")
    
    print("\nBy Accuracy:")
    best_acc = df.loc[df['accuracy'].idxmax()]
    print(f"  Architecture: {best_acc['model_label']}")
    print(f"  Embedding: {best_acc['embedding']} (Window={best_acc['window']})")
    print(f"  News: {best_acc['has_news']}")
    print(f"  Accuracy: {best_acc['accuracy']:.4f}, F1-Macro: {best_acc['f1_macro']:.4f}")
    
    # Show comparison between architectures
    print("\n📊 AVERAGE PERFORMANCE BY ARCHITECTURE (With News):")
    arch_summary = df[df['has_news'] == 'With News'].groupby('model_label')[
        ['accuracy', 'f1_macro']
    ].mean().sort_values('f1_macro', ascending=False)
    for arch, row in arch_summary.iterrows():
        print(f"  {arch:12s}: F1={row['f1_macro']:.4f}, Acc={row['accuracy']:.4f}")
    
    print("\n")


if __name__ == "__main__":
    main()

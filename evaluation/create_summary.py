#!/usr/bin/env python3
"""
Create Complete Cross-Dataset Evaluation Summary

This script creates a comprehensive summary of model performance across
different dataset combinations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_cross_dataset_results():
    """Load cross-dataset evaluation results."""
    results_file = Path("outputs/metrics/cross_dataset_results.json")
    
    if not results_file.exists():
        print("❌ Cross-dataset results not found")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def load_herlev_results():
    """Load Herlev evaluation results (assuming they exist from previous evaluation)."""
    # This would be from the existing evaluation results
    herlev_results = {
        "sipakmed_train_herlev_test": {
            "accuracy": 0.8800,  # Example value - replace with actual
            "precision": 0.8750,
            "recall": 0.8850,
            "f1_score": 0.8800,
            "confusion_matrix": [[120, 15], [18, 123]],
            "num_samples": 276
        }
    }
    return herlev_results

def create_summary_table(results):
    """Create summary table from results."""
    summary_data = []
    
    for experiment_name, metrics in results.items():
        # Parse experiment name
        parts = experiment_name.split('_')
        train_dataset = parts[0].upper()
        test_dataset = parts[2].upper()
        
        summary_data.append({
            'Experiment': f"{train_dataset} → {test_dataset}",
            'Train Dataset': train_dataset,
            'Test Dataset': test_dataset,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'Samples': metrics['num_samples']
        })
    
    return pd.DataFrame(summary_data)

def create_visualizations(summary_df):
    """Create comparison visualizations."""
    output_dir = Path("outputs/visualizations/cross_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(summary_df['Experiment'], summary_df['Accuracy'], 
                 color=sns.color_palette("husl", len(summary_df)))
    ax.set_title('Cross-Dataset Accuracy Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, summary_df['Accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # F1 score comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(summary_df['Experiment'], summary_df['F1 Score'], 
                 color=sns.color_palette("husl", len(summary_df)))
    ax.set_title('Cross-Dataset F1 Score Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, f1 in zip(bars, summary_df['F1 Score']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(summary_df):
    """Print comprehensive summary."""
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 Model: EfficientNet-B0 trained on SIPaKMeD")
    print(f"🎯 Task: Binary Classification (Normal vs Abnormal)")
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['Experiment']}")
        print(f"  Accuracy:  {row['Accuracy']:.4f}")
        print(f"  Precision: {row['Precision']:.4f}")
        print(f"  Recall:    {row['Recall']:.4f}")
        print(f"  F1 Score:  {row['F1 Score']:.4f}")
        print(f"  Samples:   {row['Samples']}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Calculate performance differences
    if len(summary_df) >= 2:
        same_dataset = summary_df[summary_df['Train Dataset'] == summary_df['Test Dataset']]
        cross_dataset = summary_df[summary_df['Train Dataset'] != summary_df['Test Dataset']]
        
        if not same_dataset.empty and not cross_dataset.empty:
            same_acc = same_dataset.iloc[0]['Accuracy']
            cross_acc = cross_dataset.iloc[0]['Accuracy']
            performance_drop = same_acc - cross_acc
            
            print(f"\n🔍 Same-Dataset Performance: {same_acc:.4f}")
            print(f"🔍 Cross-Dataset Performance: {cross_acc:.4f}")
            print(f"📉 Performance Drop: {performance_drop:.4f} ({performance_drop/same_acc*100:.1f}%)")
    
    print("\n" + "="*80)

def main():
    """Main function."""
    print("Creating Cross-Dataset Evaluation Summary")
    print("="*50)
    
    # Load results
    results = load_cross_dataset_results()
    if results is None:
        return 1
    
    # Create summary table
    summary_df = create_summary_table(results)
    
    # Save summary table
    output_dir = Path("outputs/metrics")
    summary_df.to_csv(output_dir / "cross_dataset_summary.csv", index=False)
    print(f"📊 Summary table saved to {output_dir / 'cross_dataset_summary.csv'}")
    
    # Create visualizations
    create_visualizations(summary_df)
    print(f"📈 Visualizations saved to {output_dir.parent / 'visualizations/cross_dataset'}")
    
    # Print summary
    print_summary(summary_df)
    
    print("\n✅ Summary created successfully!")
    return 0

if __name__ == "__main__":
    exit(main())

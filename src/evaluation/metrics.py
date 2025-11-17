"""Evaluation metrics for pass@k and other metrics."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt


def calculate_passk(results: List[List[bool]], k_values: List[int] = None) -> Dict[int, float]:
    """
    Calculate pass@k metrics from results.
    
    Args:
        results: List of lists, where each inner list contains boolean results for one problem
        k_values: List of k values to calculate. If None, uses range(1, max_samples+1)
        
    Returns:
        Dict mapping k to pass@k accuracy
    """
    if not results:
        return {}
    
    max_samples = max(len(r) for r in results)
    if k_values is None:
        k_values = list(range(1, max_samples + 1))
    
    passk_results = {}
    
    for k in k_values:
        if k > max_samples:
            continue
            
        total_correct = 0
        total_problems = 0
        
        for problem_results in results:
            if len(problem_results) >= k:
                # For pass@k, we succeed if any of the first k attempts is correct
                total_correct += int(any(problem_results[:k]))
                total_problems += 1
        
        if total_problems > 0:
            passk_results[k] = total_correct / total_problems
        else:
            passk_results[k] = 0.0
    
    return passk_results


def plot_passk_curve(passk_results: Dict[int, float], save_path: str = None, title: str = "Pass@k Performance"):
    """
    Plot pass@k curve.
    
    Args:
        passk_results: Dict from calculate_passk
        save_path: Path to save plot (optional)
        title: Plot title
    """
    k_values = sorted(passk_results.keys())
    accuracies = [passk_results[k] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('k (number of samples)')
    plt.ylabel('Pass@k Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max(k_values))
    plt.ylim(0, 1)
    
    # Add value labels on points
    for k, acc in zip(k_values, accuracies):
        plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_from_csv(csv_path: str, plot: bool = False, output_plot: str = None) -> Dict[int, float]:
    """
    Evaluate pass@k from a CSV file.
    
    Expected CSV format:
    - One row per problem
    - Columns: 'problem_id', 'sample_0', 'sample_1', ..., 'correct_answer'
    - Sample columns contain boolean values (0/1) for correctness
    
    Args:
        csv_path: Path to CSV file
        plot: Whether to generate plot
        output_plot: Path to save plot
        
    Returns:
        Dict mapping k to pass@k accuracy
    """
    df = pd.read_csv(csv_path)
    
    # Find sample columns (assume they're named sample_0, sample_1, etc.)
    sample_cols = [col for col in df.columns if col.startswith('sample_')]
    sample_cols = sorted(sample_cols, key=lambda x: int(x.split('_')[1]))
    
    if not sample_cols:
        raise ValueError("No sample columns found in CSV. Expected columns like 'sample_0', 'sample_1', etc.")
    
    # Extract results for each problem
    results = []
    for _, row in df.iterrows():
        problem_results = [bool(row[col]) for col in sample_cols]
        results.append(problem_results)
    
    # Calculate pass@k
    passk_results = calculate_passk(results)
    
    # Print results
    print(f"Results from {csv_path}:")
    for k in sorted(passk_results.keys()):
        print(f"Pass@{k}: {passk_results[k]:.4f}")
    
    # Plot if requested
    if plot:
        plot_title = f"Pass@k Performance ({csv_path.split('/')[-1]})"
        plot_passk_curve(passk_results, save_path=output_plot, title=plot_title)
    
    return passk_results
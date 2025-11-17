#!/usr/bin/env python3
"""
Evaluate pass@k results from generated samples.

Usage:
    python scripts/evaluate.py results/qwen_math_k16.csv --plot --output_plot plot.png
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import evaluate_from_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k results")
    
    parser.add_argument("csv_file", type=str, 
                       help="Path to CSV file with results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate pass@k plot")
    parser.add_argument("--output_plot", type=str, default=None,
                       help="Path to save plot (if not specified, displays plot)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return 1
    
    try:
        # Evaluate pass@k
        passk_results = evaluate_from_csv(
            args.csv_file, 
            plot=args.plot, 
            output_plot=args.output_plot
        )
        
        print(f"\\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main() or 0)
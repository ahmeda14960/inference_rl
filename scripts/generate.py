#!/usr/bin/env python3
"""
Generate text samples using vLLM for various benchmarks.

Usage:
    python scripts/generate.py --model qwen_math --dataset math500 --k 16 --temperature 0.8
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.loaders import load_dataset, list_available_datasets
from models.model_configs import get_model_config, list_available_models
from generation.vllm_generator import VLLMGenerator, GenerationConfig


def main():
    parser = argparse.ArgumentParser(description="Generate text samples using vLLM")
    
    # Model and dataset
    parser.add_argument("--model", type=str, 
                       help="Model name (use --list-models to see available)")
    parser.add_argument("--dataset", type=str, default="math500",
                       help="Dataset name (use --list-datasets to see available)")
    
    # Generation parameters
    parser.add_argument("--k", type=int, default=16, 
                       help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=3072,
                       help="Maximum tokens per generation")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing datasets")
    parser.add_argument("--max_problems", type=int, default=None,
                       help="Limit number of problems (for debugging)")
    parser.add_argument("--cot", action="store_true", default=True,
                       help="Use chain-of-thought prompting")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Custom output filename (without extension)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce vLLM logging output")
    
    # Hardware options
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU device to use (e.g., '0', '1', '4'). Sets CUDA_VISIBLE_DEVICES")
    
    # Utility options
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    
    args = parser.parse_args()
    
    # Handle GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # Handle utility options
    if args.list_models:
        print("Available models:")
        for name, desc in list_available_models().items():
            print(f"  {name}: {desc}")
        return
    
    if args.list_datasets:
        print("Available datasets:")
        for name, desc in list_available_datasets().items():
            print(f"  {name}: {desc}")
        return
    
    # Validate arguments
    if not args.model:
        print("Error: --model is required")
        print("Use --list-models to see available models")
        return 1
    
    try:
        model_config = get_model_config(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-models to see available models")
        return 1
    
    # Load dataset
    try:
        problems = load_dataset(args.dataset, args.data_dir, args.max_problems)
        print(f"Loaded {len(problems)} problems from {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Setup generation config
    gen_config = GenerationConfig(
        model_name=model_config.hf_model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        k=args.k,
        seed=args.seed,
        quiet=args.quiet
    )
    
    # Create generator
    generator = VLLMGenerator(gen_config)
    generator.load_model()
    
    # Generate samples
    print(f"\\nGenerating samples...")
    print(f"Model: {args.model} ({model_config.hf_model_path})")
    print(f"Dataset: {args.dataset}")
    print(f"Temperature: {args.temperature}")
    print(f"Samples per problem: {args.k}")
    print(f"Chain-of-thought: {args.cot}")
    
    results = generator.generate_math_samples(problems, args.model, cot=args.cot)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.output_name:
        output_filename = f"{args.output_name}.csv"
    else:
        output_filename = f"{args.model}_{args.dataset}_k{args.k}_temp{args.temperature}_seed{args.seed}.csv"
    
    output_path = os.path.join(args.output_dir, output_filename)
    generator.save_results(results, output_path, args.model)
    
    print(f"\\nGeneration completed successfully!")


if __name__ == "__main__":
    exit(main() or 0)
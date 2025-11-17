"""Dataset loaders for various benchmarks."""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_math500(data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load MATH500 dataset.
    
    Args:
        data_dir: Directory containing MATH500.json
        
    Returns:
        List of problem dictionaries with 'prompt' and 'answer' keys
    """
    math500_path = Path(data_dir) / "MATH500.json"
    
    if not math500_path.exists():
        raise FileNotFoundError(f"MATH500.json not found at {math500_path}")
    
    with open(math500_path, 'r') as f:
        dataset = json.load(f)
    
    # Ensure consistent format
    for item in dataset:
        if 'prompt' not in item or 'answer' not in item:
            raise ValueError("MATH500 dataset items must have 'prompt' and 'answer' keys")
    
    return dataset


def load_dataset(dataset_name: str, data_dir: str = "data", max_problems: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('math500', 'gpqa', 'humaneval')
        data_dir: Directory containing dataset files
        max_problems: Limit number of problems (for debugging)
        
    Returns:
        List of problem dictionaries
    """
    if dataset_name.lower() in ['math500', 'math']:
        dataset = load_math500(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: ['math500']")
    
    if max_problems is not None and max_problems > 0:
        dataset = dataset[:max_problems]
        print(f"Limited dataset to {len(dataset)} problems")
    
    return dataset


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    dataset_info = {
        "math500": {
            "name": "MATH500",
            "description": "500 mathematical reasoning problems from the MATH dataset",
            "tasks": "Mathematical problem solving",
            "metrics": "Pass@k accuracy",
            "file": "MATH500.json"
        }
    }
    
    name_key = dataset_name.lower()
    if name_key in dataset_info:
        return dataset_info[name_key]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def list_available_datasets() -> Dict[str, str]:
    """List all available datasets with descriptions."""
    return {
        "math500": "500 mathematical reasoning problems from the MATH dataset"
    }
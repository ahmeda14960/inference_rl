#!/usr/bin/env python3
"""
Basic functionality tests for the refactored inference_rl codebase.
Run with: python tests/test_basic_functionality.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from evaluation.parse_utils import parse_answer
        from evaluation.math_grader import grade_answer, safe_grade_math
        from evaluation.math_normalize import normalize_answer
        from evaluation.metrics import calculate_passk, plot_passk_curve
        from generation.prompts import format_math_prompt, format_gpqa_prompt
        from models.model_configs import get_model_config, list_available_models
        from datasets.loaders import load_math500, load_dataset
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_parse_utils():
    """Test answer parsing functionality."""
    print("\nTesting answer parsing...")
    
    from evaluation.parse_utils import parse_answer
    
    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("Step 1... Step 2... \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("Multiple \\boxed{wrong} answers \\boxed{correct}", "correct"),
        ("No boxed answer here", None),
        ("\\fbox{alternative}", "alternative")
    ]
    
    all_passed = True
    for input_text, expected in test_cases:
        result = parse_answer(input_text)
        if result == expected:
            print(f"✓ '{input_text[:30]}...' -> '{result}'")
        else:
            print(f"✗ '{input_text[:30]}...' -> Expected '{expected}', got '{result}'")
            all_passed = False
    
    return all_passed

def test_math_grader():
    """Test mathematical answer grading."""
    print("\nTesting math grader...")
    
    from evaluation.math_grader import grade_answer, safe_grade_math
    
    test_cases = [
        ("42", "42", True),
        ("42.0", "42", True),
        ("\\frac{1}{2}", "0.5", True),
        ("1/2", "0.5", True),
        ("wrong", "42", False),
        (None, "42", False),
        ("42", None, False)
    ]
    
    all_passed = True
    for given, correct, expected in test_cases:
        result = grade_answer(given, correct)
        safe_result = safe_grade_math(given, correct)
        
        if result == expected and safe_result == int(expected):
            print(f"✓ '{given}' vs '{correct}' -> {result}")
        else:
            print(f"✗ '{given}' vs '{correct}' -> Expected {expected}, got {result}")
            all_passed = False
    
    return all_passed

def test_model_configs():
    """Test model configuration system."""
    print("\nTesting model configs...")
    
    from models.model_configs import get_model_config, list_available_models
    
    # Test listing models
    models = list_available_models()
    if len(models) > 0:
        print(f"✓ Found {len(models)} available models")
        for name in list(models.keys())[:3]:  # Show first 3
            print(f"  - {name}: {models[name]}")
    else:
        print("✗ No models found")
        return False
    
    # Test getting specific model
    try:
        qwen_config = get_model_config("qwen_math")
        if qwen_config.name == "qwen_math" and "Qwen" in qwen_config.hf_model_path:
            print(f"✓ qwen_math config: {qwen_config.hf_model_path}")
        else:
            print(f"✗ Unexpected qwen_math config: {qwen_config}")
            return False
    except Exception as e:
        print(f"✗ Error getting qwen_math config: {e}")
        return False
    
    return True

def test_dataset_loader():
    """Test dataset loading."""
    print("\nTesting dataset loader...")
    
    from datasets.loaders import load_math500, list_available_datasets
    
    # Test listing datasets
    datasets = list_available_datasets()
    if "math500" in datasets:
        print(f"✓ Found math500 dataset: {datasets['math500']}")
    else:
        print("✗ math500 dataset not found")
        return False
    
    # Test loading MATH500
    try:
        data_dir = Path(__file__).parent.parent / "data"
        if not (data_dir / "MATH500.json").exists():
            print("✗ MATH500.json not found, skipping load test")
            return True
        
        problems = load_math500(str(data_dir))
        if len(problems) > 0:
            print(f"✓ Loaded {len(problems)} problems from MATH500")
            # Check first problem structure
            first = problems[0]
            if "prompt" in first and "answer" in first:
                print(f"✓ Problem structure valid: {list(first.keys())}")
            else:
                print(f"✗ Invalid problem structure: {list(first.keys())}")
                return False
        else:
            print("✗ No problems loaded")
            return False
    except Exception as e:
        print(f"✗ Error loading MATH500: {e}")
        return False
    
    return True

def test_prompts():
    """Test prompt formatting."""
    print("\nTesting prompt formatting...")
    
    from generation.prompts import format_math_prompt, format_gpqa_prompt
    
    # Test math prompt
    question = "What is 2 + 2?"
    prompt = format_math_prompt(question, "qwen_math", cot=True)
    
    if "solve the following math problem" in prompt.lower() and "step by step" in prompt.lower():
        print("✓ Math prompt formatting works")
    else:
        print(f"✗ Unexpected math prompt: {prompt}")
        return False
    
    # Test GPQA prompt
    choices = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
    gpqa_prompt = format_gpqa_prompt("Test question?", choices)
    
    if "multiple choice" in gpqa_prompt.lower() and "A) Option A" in gpqa_prompt:
        print("✓ GPQA prompt formatting works")
    else:
        print(f"✗ Unexpected GPQA prompt: {gpqa_prompt}")
        return False
    
    return True

def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics...")
    
    from evaluation.metrics import calculate_passk
    
    # Test data: 3 problems, each with 4 samples
    test_results = [
        [True, False, False, False],   # Problem 1: correct on first try
        [False, False, True, False],   # Problem 2: correct on third try
        [False, False, False, False],  # Problem 3: all wrong
    ]
    
    passk = calculate_passk(test_results, k_values=[1, 2, 3, 4])
    
    # Expected: pass@1 = 1/3, pass@2 = 1/3, pass@3 = 2/3, pass@4 = 2/3
    expected = {1: 1/3, 2: 1/3, 3: 2/3, 4: 2/3}
    
    all_passed = True
    for k, expected_val in expected.items():
        if abs(passk[k] - expected_val) < 1e-6:
            print(f"✓ Pass@{k}: {passk[k]:.3f}")
        else:
            print(f"✗ Pass@{k}: Expected {expected_val:.3f}, got {passk[k]:.3f}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("=" * 50)
    print("INFERENCE_RL FUNCTIONALITY TESTS")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_parse_utils,
        test_math_grader,
        test_model_configs,
        test_dataset_loader,
        test_prompts,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored codebase is working correctly.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
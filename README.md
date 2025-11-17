# Inference RL - Clean Mathematical Reasoning Benchmark

A clean, focused codebase for running vLLM inference and evaluation on mathematical reasoning benchmarks.

## Features

- **Clean vLLM Integration**: Efficient batched generation with proper error handling
- **Modular Design**: Separate modules for models, datasets, generation, and evaluation  
- **Pass@k Evaluation**: Built-in support for pass@k metrics and plotting
- **Multiple Benchmarks**: Support for MATH500 and extensible to other datasets
- **Simple CLI**: Easy-to-use command line interface

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd inference_rl

# Install dependencies
pip install -e .

# Or install with uv (recommended)
uv pip install -e .
```

## Quick Start

### Generate Samples

```bash
# Generate 16 samples per problem with Qwen Math model on GPU 4
python scripts/generate.py \\
    --gpu 4 \\
    --model qwen_math \\
    --dataset math500 \\
    --k 16 \\
    --temperature 0.8 \\
    --output_dir results

# Alternative: use environment setup
source env.sh  # Sets CUDA_VISIBLE_DEVICES=4
python scripts/generate.py --model qwen_math --dataset math500 --k 16
```

### Evaluate Results

```bash
# Evaluate and plot pass@k curves
python scripts/evaluate.py results/qwen_math_math500_k16_temp0.8_seed0.csv \\
    --plot \\
    --output_plot results/qwen_math_passk_plot.png
```

## Available Models

- `qwen`: Qwen 2.5 7B base model
- `qwen_math`: Qwen 2.5 Math 7B (specialized for mathematics)  
- `qwen_math_grpo`: Qwen Math 7B with GRPO fine-tuning
- `phi`: Microsoft Phi-3.5 Mini Instruct
- `tulu`: Llama 3.1 Tulu 3 8B with DPO

List all models: `python scripts/generate.py --list-models`

## Available Datasets

- `math500`: 500 mathematical reasoning problems from the MATH dataset

List all datasets: `python scripts/generate.py --list-datasets`

## Project Structure

```
inference_rl/
├── src/
│   ├── datasets/           # Dataset loading utilities
│   ├── evaluation/         # Grading and metrics  
│   ├── generation/         # vLLM generation code
│   └── models/            # Model configurations
├── data/
│   └── MATH500.json       # Core dataset
├── scripts/
│   ├── generate.py        # Generation CLI
│   └── evaluate.py        # Evaluation CLI
├── pyproject.toml         # Dependencies and build config
└── README.md
```

## Advanced Usage

### Custom Models

Add custom models programmatically:

```python
from src.models.model_configs import add_custom_model

add_custom_model(
    name="my_model",
    hf_model_path="path/to/model",
    is_chat_model=False,
    description="My custom model"
)
```

### Debugging with Fewer Problems

```bash
python scripts/generate.py \\
    --model qwen_math \\
    --dataset math500 \\
    --max_problems 10 \\
    --k 4
```

### Quiet Mode (Reduce Logging)

```bash
python scripts/generate.py \\
    --model qwen_math \\
    --dataset math500 \\
    --quiet
```

## Output Format

Generated CSV files contain:
- Problem metadata (question, correct_answer)
- Individual completions and parsed answers  
- Correctness flags for each sample
- Columns named `sample_0`, `sample_1`, etc. for easy pass@k evaluation

## Dependencies

- vLLM (for efficient inference)
- PyTorch and Transformers
- SymPy (for mathematical answer grading)
- Pandas, NumPy, Matplotlib (for data handling and visualization)

See `pyproject.toml` for complete dependency list.
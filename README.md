# Inference RL - Reasoning with Sampling

This repository contains experiments on reasoning with sampling for language models.

## Installation Instructions

### Prerequisites

- Conda/Miniconda installed
- CUDA-capable GPU (recommended for running LLM experiments)
- Git

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd inference_rl
```

2. Navigate to the reasoning-with-sampling directory:
```bash
cd reasoning-with-sampling
```

3. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate psamp
```

This will install all required dependencies including:
- PyTorch with CUDA 12.4 support
- vLLM for efficient LLM inference
- Transformers, tokenizers, and other ML libraries
- Ray for distributed computing
- Various evaluation libraries (AlpacaEval, etc.)

### Verifying Installation

To verify the installation was successful, you can run:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Environment Details

The `psamp` environment includes:
- **Deep Learning**: PyTorch 2.8.0, torchvision, transformers 4.47.1
- **Inference**: vLLM 0.6.6.post1, outlines
- **Evaluation**: alpaca-eval, datasets
- **General ML**: numpy, scipy, scikit-learn, pandas
- **Visualization**: matplotlib, manim

## Project Structure

```
reasoning-with-sampling/
├── environment.yml           # Conda environment specification
├── README.md                 # Project documentation
├── toy_composition.py        # Toy composition experiments
├── llm_experiments/          # Main experiment code
│   ├── power_samp_math.py   # Power sampling for MATH500
│   ├── power_samp_he.py     # Power sampling for HumanEval
│   ├── power_samp_gpqa.py   # Power sampling for GPQA Diamond
│   ├── power_samp_alpaca.py # Power sampling for AlpacaEval 2.0
│   ├── eval_*.py            # Evaluation scripts
│   ├── passk_*.py           # Pass@k performance scripts
│   ├── grader_utils/        # Grading utilities
│   ├── scripts/             # SLURM batch scripts
│   └── data/                # Dataset files (MATH500.json included)
└── teaser.png               # Project teaser image
```

## Quick Start

For detailed usage instructions on running experiments and evaluations, see the [reasoning-with-sampling README](reasoning-with-sampling/README.md).

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors, ensure your NVIDIA drivers are up to date and compatible with CUDA 12.4.

### Memory Issues
vLLM and large language models require significant GPU memory. Adjust batch sizes or model sizes if you encounter OOM errors.

### Conda Environment Conflicts
If you have issues creating the environment, try:
```bash
conda clean --all
conda env create -f reasoning-with-sampling/environment.yml
```

## Citation

If you use this code, please cite:

```
Reasoning with Sampling: Your Base Model is Smarter Than You Think
Aayush Karan, Yilun Du
Harvard
```

Paper: https://arxiv.org/abs/2510.14901
Project Page: https://aakaran.github.io/reasoning_with_sampling/

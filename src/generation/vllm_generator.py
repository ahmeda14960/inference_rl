"""vLLM-based text generation."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from vllm import LLM, SamplingParams

from evaluation.parse_utils import parse_answer
from evaluation.math_grader import safe_grade_math
from generation.prompts import format_math_prompt


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model_name: str
    temperature: float = 0.8
    max_tokens: int = 3072
    k: int = 16  # Number of samples per problem
    seed: int = 0
    quiet: bool = False


class VLLMGenerator:
    """vLLM-based text generator."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.llm = None
        self.tokenizer = None
        
        if config.quiet:
            self._setup_quiet_logging()
    
    def _setup_quiet_logging(self):
        """Reduce vLLM logging output."""
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        os.environ.setdefault("VLLM_DISABLE_LOG_STATS", "1")
        try:
            logging.getLogger("vllm").setLevel(logging.WARNING)
        except Exception:
            pass
    
    def load_model(self):
        """Load the vLLM model."""
        print(f"Loading model: {self.config.model_name}")
        
        if self.config.quiet:
            import io
            from contextlib import redirect_stdout, redirect_stderr
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                self.llm = LLM(model=self.config.model_name, trust_remote_code=True)
                self.tokenizer = self.llm.get_tokenizer()
        else:
            self.llm = LLM(model=self.config.model_name, trust_remote_code=True)
            self.tokenizer = self.llm.get_tokenizer()
        
        print("Model loaded successfully")
    
    def generate_math_samples(self, problems: List[Dict[str, Any]], model_short_name: str, cot: bool = True) -> List[Dict[str, Any]]:
        """
        Generate samples for math problems.
        
        Args:
            problems: List of problem dicts with 'prompt' and 'answer' keys
            model_short_name: Short name for prompt formatting
            cot: Whether to use chain-of-thought prompting
            
        Returns:
            List of result dicts with completions and evaluations
        """
        if self.llm is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            n=self.config.k,  # Generate k samples per prompt
        )
        
        # Prepare all prompts
        all_prompts = []
        all_metadata = []
        
        for i, problem in enumerate(problems):
            question = problem["prompt"]
            answer = problem["answer"]
            
            prompt = format_math_prompt(question, model_short_name, self.tokenizer, cot)
            all_prompts.append(prompt)
            all_metadata.append({
                "problem_idx": i,
                "question": question,
                "correct_answer": answer
            })
        
        print(f"Generating {self.config.k} samples for {len(all_prompts)} problems...")
        
        # Generate responses
        if self.config.quiet:
            import io
            from contextlib import redirect_stdout, redirect_stderr
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                outputs = self.llm.generate(all_prompts, sampling_params)
        else:
            outputs = self.llm.generate(all_prompts, sampling_params)
        
        # Process outputs
        results = []
        for output, metadata in tqdm(zip(outputs, all_metadata), total=len(outputs), desc="Processing results"):
            question = metadata["question"]
            correct_answer = metadata["correct_answer"]
            
            # Extract all k completions for this problem
            completions = [o.text for o in output.outputs]
            answers = [parse_answer(completion) for completion in completions]
            
            # Compute correctness for each sample
            correct_flags = [safe_grade_math(ans, correct_answer) for ans in answers]
            
            # Store result
            result = {
                "problem_idx": metadata["problem_idx"],
                "question": question,
                "correct_answer": correct_answer,
                "completions": completions,
                "parsed_answers": answers,
                "correct_flags": correct_flags,
                "k": self.config.k
            }
            
            # Add individual sample columns for easier analysis
            for j, (completion, answer, correct) in enumerate(zip(completions, answers, correct_flags)):
                result[f"completion_{j}"] = completion
                result[f"answer_{j}"] = answer
                result[f"sample_{j}"] = correct
            
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str, model_short_name: str) -> None:
        """Save results to CSV file."""
        df = pd.DataFrame(results)
        
        # Calculate overall statistics
        total_problems = len(results)
        sample_cols = [col for col in df.columns if col.startswith('sample_')]
        
        # Calculate pass@k for different k values
        passk_stats = {}
        for k in range(1, len(sample_cols) + 1):
            k_cols = sample_cols[:k]
            # Pass@k: any of the first k samples is correct
            passk_success = df[k_cols].any(axis=1).sum()
            passk_stats[f"pass@{k}"] = passk_success / total_problems
        
        # Print statistics
        print(f"\nResults Summary:")
        print(f"Total problems: {total_problems}")
        print(f"Model: {model_short_name}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Samples per problem: {self.config.k}")
        print("\nPass@k results:")
        for metric, value in passk_stats.items():
            print(f"{metric}: {value:.4f}")
        
        # Save CSV
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Save summary JSON
        summary_path = output_path.replace('.csv', '_summary.json')
        summary = {
            "model": model_short_name,
            "temperature": self.config.temperature,
            "k": self.config.k,
            "total_problems": total_problems,
            **passk_stats
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")
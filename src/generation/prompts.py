"""Prompt templates for different tasks."""

# Math prompts
MATH_PROMPT = "Can you solve the following math problem? "
MATH_BASE = " Put your final answer within \\boxed{{}}."
MATH_COT = " Please reason step by step, and put your final answer within \\boxed{{}}."
MATH_COT_ALT = " Please explain your reasoning with a detailed, step-by-step solution, and present your final answer within \\boxed{{}}."

# GPQA prompts
GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). "
    "Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def format_math_prompt(question: str, model: str, tokenizer=None, cot: bool = True) -> str:
    """
    Format a math question into a prompt for the given model.
    
    Args:
        question: The math question
        model: Model identifier 
        tokenizer: Model tokenizer (needed for chat models)
        cot: Whether to use chain-of-thought prompting
        
    Returns:
        Formatted prompt string
    """
    content_str = MATH_PROMPT + question
    if cot:
        content_str += MATH_COT
    else:
        content_str += MATH_BASE
    
    # For chat models that need special formatting
    if model in ["qwen_math_grpo", "phi_grpo"] and tokenizer is not None:
        messages = [{"role": "user", "content": content_str}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return content_str


def format_gpqa_prompt(question: str, choices: dict) -> str:
    """
    Format a GPQA question into a prompt.
    
    Args:
        question: The question text
        choices: Dict with keys A, B, C, D
        
    Returns:
        Formatted prompt string
    """
    return GPQA_QUERY_TEMPLATE.format(
        Question=question,
        A=choices['A'],
        B=choices['B'], 
        C=choices['C'],
        D=choices['D']
    )
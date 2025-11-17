"""Utilities for parsing answers from model outputs."""

import json
import numpy as np


def remove_boxed(s):
    """Remove \\boxed{} or \\fbox{} wrapper from answer."""
    if s is None:
        return None
    
    # Try \\boxed{ first
    left_boxed = "\\boxed{"
    if s.startswith(left_boxed) and s.endswith("}"):
        return s[len(left_boxed):-1]
    
    # Try \\fbox{ 
    left_fbox = "\\fbox{"
    if s.startswith(left_fbox) and s.endswith("}"):
        return s[len(left_fbox):-1]
    
    return None


def last_boxed_only_string(string):
    """Extract the last \\boxed{...} or \\fbox{...} element from string."""
    # Check for both \\boxed and \\fbox, take the latest one
    boxed_idx = string.rfind("\\boxed")
    fbox_idx = string.rfind("\\fbox")
    
    # Use the rightmost (latest) occurrence
    idx = max(boxed_idx, fbox_idx)
    if idx < 0:
        return None

    # Find the opening brace after the command
    i = idx
    while i < len(string) and string[i] != '{':
        i += 1
    
    if i >= len(string):
        return None
        
    # Now count braces starting from the opening brace
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx is None:
        return None
    else:
        return string[idx:right_brace_idx + 1]


def parse_answer(input_str):
    """Parse answer from model output by extracting content from \\boxed{}."""
    return remove_boxed(last_boxed_only_string(input_str))
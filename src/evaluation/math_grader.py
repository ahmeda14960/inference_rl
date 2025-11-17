"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from evaluation.math_normalize import normalize_answer


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _sympy_equal(expr1: str, expr2: str) -> bool:
    """Check if two expressions are equal using sympy."""
    try:
        sympy_expr1 = _sympy_parse(expr1)
        sympy_expr2 = _sympy_parse(expr2)
        diff = sympy_expr1 - sympy_expr2
        return sympy.simplify(diff) == 0
    except:
        return False


def _contains_bad_substrings(expr: str) -> bool:
    """Check if expression contains problematic substrings."""
    for bad_substring in BAD_SUBSTRINGS:
        if bad_substring in expr:
            return True
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr):
            return True
    return False


def _clean_units(expr: str) -> str:
    """Remove units from expression."""
    # Remove common units
    units = ["cm", "mm", "m", "km", "kg", "g", "s", "min", "hour", "°", "degrees"]
    for unit in units:
        expr = re.sub(r'\b' + unit + r'\b', '', expr)
    return expr.strip()


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    Grade a given answer against ground truth.
    
    Args:
        given_answer: Student/model answer
        ground_truth: Correct answer
        
    Returns:
        bool: True if correct, False otherwise
    """
    if given_answer is None or ground_truth is None:
        return False
        
    # Normalize both answers
    given_answer = normalize_answer(given_answer)
    ground_truth = normalize_answer(ground_truth)
    
    if given_answer is None or ground_truth is None:
        return False
    
    # Direct string match
    if given_answer == ground_truth:
        return True
    
    # Clean and check for bad substrings
    if _contains_bad_substrings(given_answer) or _contains_bad_substrings(ground_truth):
        return False
    
    # Clean units
    given_clean = _clean_units(given_answer)
    ground_clean = _clean_units(ground_truth)
    
    # Try sympy comparison
    if _sympy_equal(given_clean, ground_clean):
        return True
    
    return False


def safe_grade_math(ans, correct_ans):
    """Safe wrapper around grade_answer that returns 0 on exception."""
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0
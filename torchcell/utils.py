import os.path as osp

def format_scientific_notation(num: float) -> str:
    """Format a number into concise scientific notation."""
    # Format with 1 significant digit after the decimal, if necessary
    sci_notation = f"{num:.1e}"
    # Remove unnecessary parts for concise representation
    return sci_notation.replace('.0e', 'e').replace('+', '')

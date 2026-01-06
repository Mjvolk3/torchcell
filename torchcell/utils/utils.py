import math


def format_scientific_notation(num: float) -> str:
    """Format a number into scientific notation, preserving all significant digits.

    Examples:
        10000 -> "1e04"
        10007 -> "1.0007e04"
        1234567 -> "1.234567e06"
        100 -> "1e02"
    """
    if num == 0:
        return "0"

    # Handle negative numbers
    sign = ""
    if num < 0:
        sign = "-"
        num = abs(num)

    # Get exponent (power of 10)
    exponent = int(math.floor(math.log10(num)))
    mantissa = num / (10**exponent)

    # Find minimum precision needed to preserve the exact value
    mantissa_str = f"{mantissa:.0f}"  # Default for Pylance
    for precision in range(15):
        mantissa_str = f"{mantissa:.{precision}f}"
        reconstructed = float(mantissa_str) * (10**exponent)
        if round(reconstructed) == round(num):
            break

    # Clean up: remove trailing zeros and unnecessary decimal point
    mantissa_str = mantissa_str.rstrip("0").rstrip(".")

    return f"{sign}{mantissa_str}e{exponent:02d}"
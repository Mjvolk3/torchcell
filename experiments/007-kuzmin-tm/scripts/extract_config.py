#!/usr/bin/env python3
"""
Extract configuration from glucose_oxygen_sensitivity_all_media.py
"""

import sys
import re

def extract_config(script_path):
    """Extract CONDITIONS dict from the Python script."""
    with open(script_path, 'r') as f:
        content = f.read()

    # Find the CONDITIONS dict
    pattern = r"CONDITIONS\s*=\s*\{([^}]+)\}"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Error: Could not find CONDITIONS dict", file=sys.stderr)
        sys.exit(1)

    config_str = match.group(1)

    # Extract media types
    media_match = re.search(r"'media_types':\s*\[([^\]]+)\]", config_str)
    if media_match:
        media_types = [m.strip().strip("'\"") for m in media_match.group(1).split(',')]
        print(' '.join(media_types))

    # Extract glucose levels
    glucose_match = re.search(r"'glucose_levels':\s*\[([^\]]+)\]", config_str)
    if glucose_match:
        glucose_levels = [g.strip() for g in glucose_match.group(1).split(',')
                         if g.strip() and not g.strip().startswith('#')]
        print(' '.join(glucose_levels))

    # Extract oxygen levels
    oxygen_match = re.search(r"'oxygen_levels':\s*\[([^\]]+)\]", config_str)
    if oxygen_match:
        oxygen_levels = [o.strip() for o in oxygen_match.group(1).split(',')
                        if o.strip() and not o.strip().startswith('#')]
        print(' '.join(oxygen_levels))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: extract_config.py <script_path>", file=sys.stderr)
        sys.exit(1)

    extract_config(sys.argv[1])

# scripts/generate_calmorph_labels
# [[scripts.generate_calmorph_labels]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/generate_calmorph_labels
# Test file: scripts/test_generate_calmorph_labels.py

"""
Generate calmorph_labels.py from SI_1_parameters.xlsx.

This script extracts CalMorph parameter labels from the manually prepared Excel file
and generates the calmorph_labels.py module with all 501 morphological parameters.

Usage:
    python scripts/generate_calmorph_labels.py
"""

import pandas as pd
from pathlib import Path
import requests
import tempfile


def generate_calmorph_labels():
    """Extract CalMorph labels from Excel and generate Python module."""

    # Download Excel file from Box URL
    excel_url = (
        "https://uofi.box.com/shared/static/jf6hdqhclc1wen42rv1kny6ztl0xjovr.xlsx"
    )

    print(f"Downloading Excel file from: {excel_url}")

    # Download to temporary file
    response = requests.get(excel_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    # Read Excel file
    df = pd.read_excel(tmp_path)

    print(f"Loaded Excel file with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Extract labels
    labels = {}
    for _, row in df.iterrows():
        # Get ID and description using correct column names
        param_id = str(row.get("ID", "")).strip() if pd.notna(row.get("ID")) else ""
        description = (
            str(row.get("Description", "")).strip()
            if pd.notna(row.get("Description"))
            else ""
        )

        # Skip empty or invalid entries
        if param_id and description and param_id not in ["nan", "", "ID"]:
            # Replace spaces with underscores for consistency
            param_id = param_id.replace(" ", "_")
            description = description.replace(" ", "_")
            labels[param_id] = description

    print(f"Extracted {len(labels)} valid parameters")

    # Sort labels
    sorted_labels = dict(sorted(labels.items()))

    # Group by category
    categories = {
        "C": [],  # Cell morphology
        "A": [],  # Actin organization
        "D": [],  # Nuclear morphology
        "CCV": [],  # Cell morphology CV
        "ACV": [],  # Actin CV
        "DCV": [],  # Nuclear CV
        "TCV": [],  # Other CV
    }

    for key, value in sorted_labels.items():
        if key.startswith("CCV"):
            categories["CCV"].append((key, value))
        elif key.startswith("ACV"):
            categories["ACV"].append((key, value))
        elif key.startswith("DCV"):
            categories["DCV"].append((key, value))
        elif key.startswith("TCV"):
            categories["TCV"].append((key, value))
        elif key.startswith("C"):
            categories["C"].append((key, value))
        elif key.startswith("A"):
            categories["A"].append((key, value))
        elif key.startswith("D"):
            categories["D"].append((key, value))

    # Separate base parameters from CV parameters
    base_labels = {}
    cv_labels = {}
    
    for key, value in sorted_labels.items():
        if key.startswith(('CCV', 'ACV', 'DCV', 'TCV')):
            cv_labels[key] = value
        else:
            base_labels[key] = value
    
    # Generate Python module content
    output = []
    output.append('"""')
    output.append("CalMorph parameter labels mapping.")
    output.append(
        "Extracted from Ohya et al. 2005 supplementary information (SI_1_parameters.xlsx)."
    )
    output.append(
        f"Total of {len(sorted_labels)} morphological parameters:"
    )
    output.append(f"  - {len(base_labels)} base parameters (CALMORPH_LABELS)")
    output.append(f"  - {len(cv_labels)} coefficient of variation parameters (CALMORPH_STATISTICS)")
    output.append('"""')
    output.append("")
    output.append("from typing import Dict")
    output.append("")
    output.append("# All CalMorph parameters (501 total)")
    output.append("# This is the complete set of all parameters from the original dataset")
    output.append(f"CALMORPH_PARAMETERS: Dict[str, str] = {{")

    # Add all parameters first (CALMORPH_PARAMETERS)
    first = True
    category_info = [
        ("C", "Cell morphology parameters"),
        ("A", "Actin organization parameters"),
        ("D", "Nuclear morphology parameters"),
        ("CCV", "Cell morphology coefficient of variation"),
        ("ACV", "Actin coefficient of variation"),
        ("DCV", "Nuclear coefficient of variation"),
        ("TCV", "Other coefficient of variation"),
    ]

    for category_key, category_name in category_info:
        if categories[category_key]:
            if not first:
                output.append("")
            output.append(f"    # {category_name} ({category_key})")
            for param_id, description in categories[category_key]:
                # Escape quotes in description
                description = description.replace('"', '\\"')
                output.append(f'    "{param_id}": "{description}",')
            first = False

    # Remove trailing comma from last entry
    if output[-1].endswith(","):
        output[-1] = output[-1][:-1]
    
    output.append("}")
    output.append("")
    output.append("")
    
    # Add base parameters (CALMORPH_LABELS)
    output.append(f"# Base CalMorph parameters ({len(base_labels)} parameters)")
    output.append("# These are the primary morphological measurements (excludes CV parameters)")
    output.append("CALMORPH_LABELS: Dict[str, str] = {")
    
    first = True
    for category_key, category_name in [
        ("C", "Cell morphology parameters"),
        ("A", "Actin organization parameters"),
        ("D", "Nuclear morphology parameters"),
    ]:
        params = [(k, v) for k, v in categories[category_key] if k in base_labels]
        if params:
            if not first:
                output.append("")
            output.append(f"    # {category_name} ({category_key})")
            for param_id, description in params:
                description = description.replace('"', '\\"')
                output.append(f'    "{param_id}": "{description}",')
            first = False
    
    if output[-1].endswith(","):
        output[-1] = output[-1][:-1]
    
    output.append("}")
    output.append("")
    output.append("")
    
    # Add CV parameters (CALMORPH_STATISTICS)
    output.append(f"# Coefficient of variation parameters ({len(cv_labels)} parameters)")
    output.append("# These are statistical measures of variability for base parameters")
    output.append("CALMORPH_STATISTICS: Dict[str, str] = {")
    
    first = True
    for category_key, category_name in [
        ("CCV", "Cell morphology coefficient of variation"),
        ("ACV", "Actin coefficient of variation"),
        ("DCV", "Nuclear coefficient of variation"),
        ("TCV", "Other coefficient of variation"),
    ]:
        params = [(k, v) for k, v in categories[category_key] if k in cv_labels]
        if params:
            if not first:
                output.append("")
            output.append(f"    # {category_name} ({category_key})")
            for param_id, description in params:
                description = description.replace('"', '\\"')
                output.append(f'    "{param_id}": "{description}",')
            first = False
    
    if output[-1].endswith(","):
        output[-1] = output[-1][:-1]

    output.append("}")
    output.append("")

    # Write to output file - using relative path
    output_path = Path("torchcell/datamodels/calmorph_labels.py")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and ask for confirmation
    if output_path.exists():
        response = input(f"\n{output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != "y":
            print("Aborted. File not modified.")
            return None

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    print(f"\nGenerated: {output_path}")
    print(f"Total parameters: {len(sorted_labels)}")
    print(f"  - Base parameters (CALMORPH_LABELS): {len(base_labels)}")
    print(f"  - CV parameters (CALMORPH_STATISTICS): {len(cv_labels)}")
    print(f"  - All parameters (CALMORPH_PARAMETERS): {len(sorted_labels)}")

    # Print category breakdown
    print("\nParameters by category:")
    for category_key, category_name in category_info:
        if categories[category_key]:
            base_count = sum(1 for k, _ in categories[category_key] if k in base_labels)
            cv_count = sum(1 for k, _ in categories[category_key] if k in cv_labels)
            total = len(categories[category_key])
            print(
                f"  {category_key:4} ({category_name:30}): {total:3} total ({base_count:3} base, {cv_count:3} CV)"
            )

    # Clean up temporary file
    Path(tmp_path).unlink()

    return sorted_labels


if __name__ == "__main__":
    labels = generate_calmorph_labels()
    print("\nGeneration complete!")

# torchcell/database/biocypher_out_combine
# [[torchcell.database.biocypher_out_combine]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/database/biocypher_out_combine
# Test file: tests/torchcell/database/test_biocypher_out_combine.py

import os
import shutil
import yaml
from datetime import datetime
import argparse
from deepdiff import DeepDiff
import glob
import subprocess


def merge_dicts(dict1, dict2):
    """Merge two dictionaries, preferring values from dict2 for common keys."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            elif key in ["present_in_knowledge_graph", "is_relationship"]:
                result[key] = result[key] or value
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = list(set(result[key] + value))  # Remove duplicates
            elif result[key] != value:
                raise ValueError(f"Conflict in key '{key}': {result[key]} != {value}")
        else:
            result[key] = value
    return result


def check_yaml_compatibility(yaml_files):
    combined_data = {}
    for file in yaml_files:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
        try:
            combined_data = merge_dicts(combined_data, data)
        except ValueError as e:
            raise ValueError(f"Conflict in {file}: {str(e)}")
    return combined_data


def remove_duplicates(data):
    """Remove duplicate entries from the merged YAML data."""
    if isinstance(data, dict):
        return {k: remove_duplicates(v) for k, v in data.items()}
    elif isinstance(data, list):
        return list(dict.fromkeys(data))  # Remove duplicates while preserving order
    else:
        return data


def combine_csv_files(input_dirs, output_dir):
    file_counters = {}
    for input_dir in input_dirs:
        for file in os.listdir(input_dir):
            if file.endswith(".csv"):
                base_name = file.split("-")[0]
                if file.endswith("-header.csv"):
                    shutil.copy(
                        os.path.join(input_dir, file), os.path.join(output_dir, file)
                    )
                else:
                    counter = file_counters.get(base_name, 0)
                    new_name = f"{base_name}-part{counter:03d}.csv"
                    shutil.copy(
                        os.path.join(input_dir, file),
                        os.path.join(output_dir, new_name),
                    )
                    file_counters[base_name] = counter + 1


def generate_neo4j_import_script(output_dir, neo4j_config):
    script_content = [
        "#!/bin/bash",
        "version=$(bin/neo4j-admin --version | cut -d '.' -f 1)",
        "if [[ $version -ge 5 ]]; then",
        f'    {neo4j_config["import_call_bin_prefix"]}neo4j-admin database import full \\',
        f'    --delimiter="{neo4j_config["delimiter"]}" \\',
        f'    --array-delimiter="{neo4j_config["array_delimiter"]}" \\',
        '    --quote="\'" \\',
        f'    --overwrite-destination={str(neo4j_config["wipe"]).lower()} \\',
        f'    --skip-bad-relationships={str(neo4j_config["skip_bad_relationships"]).lower()} \\',
        f'    --skip-duplicate-nodes={str(neo4j_config["skip_duplicate_nodes"]).lower()} \\',
    ]

    output_dir_name = os.path.basename(output_dir)

    # Add nodes
    for file in os.listdir(output_dir):
        if file.endswith("-header.csv") and not file.startswith(
            ("Schema_info", "Mentions")
        ):
            base_name = file.split("-")[0]
            script_content.append(
                f'    --nodes="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part*" \\'
            )

    # Add relationships
    for file in os.listdir(output_dir):
        if file.endswith("-header.csv") and (
            file.startswith(("Mentions", "MemberOf")) or "Of-header" in file
        ):
            base_name = file.split("-")[0]
            script_content.append(
                f'    --relationships="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part*" \\'
            )

    # Add database name for Neo4j 5+
    script_content[-1] = script_content[-1] + f'{neo4j_config["database_name"]}'

    # Add Neo4j 4.x version command
    script_content.extend(
        [
            "else",
            f'    {neo4j_config["import_call_bin_prefix"]}neo4j-admin import \\',
            f'    --delimiter="{neo4j_config["delimiter"]}" \\',
            f'    --array-delimiter="{neo4j_config["array_delimiter"]}" \\',
            '    --quote="\'" \\',
            f'    --force={str(neo4j_config["wipe"]).lower()} \\',
            f'    --skip-bad-relationships={str(neo4j_config["skip_bad_relationships"]).lower()} \\',
            f'    --skip-duplicate-nodes={str(neo4j_config["skip_duplicate_nodes"]).lower()} \\',
        ]
    )

    # Add nodes for Neo4j 4.x
    for file in os.listdir(output_dir):
        if file.endswith("-header.csv") and not file.startswith(
            ("Schema_info", "Mentions")
        ):
            base_name = file.split("-")[0]
            script_content.append(
                f'    --nodes="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part*" \\'
            )

    # Add relationships for Neo4j 4.x
    for file in os.listdir(output_dir):
        if file.endswith("-header.csv") and (
            file.startswith(("Mentions", "MemberOf")) or "Of-header" in file
        ):
            base_name = file.split("-")[0]
            script_content.append(
                f'    --relationships="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part*" \\'
            )

    # Add database name for Neo4j 4.x
    script_content[-1] = (
        script_content[-1] + f'--database={neo4j_config["database_name"]}'
    )

    # Close the if-else statement
    script_content.append("fi")

    with open(os.path.join(output_dir, "neo4j-admin-import-call.sh"), "w") as f:
        f.write("\n".join(script_content))


def main(input_dirs, output_base_dir, neo4j_yaml):
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_base_dir, f"{timestamp}_combined")
    os.makedirs(output_dir, exist_ok=True)

    # Check YAML compatibility
    yaml_files = [os.path.join(d, "schema_info.yaml") for d in input_dirs]
    try:
        combined_yaml = check_yaml_compatibility(yaml_files)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Write combined YAML
    with open(os.path.join(output_dir, "schema_info.yaml"), "w") as f:
        yaml.dump(combined_yaml, f)

    # Combine CSV files
    combine_csv_files(input_dirs, output_dir)

    # Load Neo4j configuration
    with open(neo4j_yaml, "r") as f:
        neo4j_config = yaml.safe_load(f)["neo4j"]

    # Generate Neo4j import script
    generate_neo4j_import_script(output_dir, neo4j_config)

    print(f"Data combined successfully in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine BioCypher output directories")
    parser.add_argument(
        "input_dirs", nargs="+", help="Input directories containing BioCypher output"
    )
    parser.add_argument(
        "--output_base_dir",
        required=True,
        help="Base output directory for combined data",
    )
    parser.add_argument(
        "--neo4j_yaml", required=True, help="Path to the Neo4j configuration YAML file"
    )
    args = parser.parse_args()

    main(args.input_dirs, args.output_base_dir, args.neo4j_yaml)

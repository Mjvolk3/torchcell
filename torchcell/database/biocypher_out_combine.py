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
import json
import csv


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


def create_schema_info_csv(output_dir, schema_yaml):
    """Create a new Schema_info CSV file based on the combined schema YAML."""
    csv_header = ":ID\tschema_info\tid\tpreferred_id\t:LABEL"

    # Ensure is_schema_info: true is within the schema_yaml
    schema_yaml["is_schema_info"] = True

    # Convert the schema to a JSON string
    schema_json = json.dumps(schema_yaml, separators=(",", ":"))

    # Create the CSV data row
    csv_data = f"Schema_info\t'{schema_json}'\tschema_info\tid\tSchema_info"

    # Write the header CSV file
    with open(os.path.join(output_dir, "Schema_info-header.csv"), "w", newline="") as f:
        f.write(csv_header + "\n")

    # Write the data CSV file
    with open(
        os.path.join(output_dir, "Schema_info-part000.csv"), "w", newline=""
    ) as f:
        f.write(csv_data + "\n")

    print(f"Created Schema_info CSV files in {output_dir}")


def combine_csv_files(input_dirs, output_dir):
    file_counters = {}
    for input_dir in input_dirs:
        for file in os.listdir(input_dir):
            if file.endswith(".csv") and not file.startswith("Schema_info"):
                base_name = file.split("-")[0]
                if file.endswith("-header.csv"):
                    if not os.path.exists(os.path.join(output_dir, file)):
                        shutil.copy(
                            os.path.join(input_dir, file),
                            os.path.join(output_dir, file),
                        )
                else:
                    counter = file_counters.get(base_name, 0)
                    new_name = f"{base_name}-part{counter:03d}.csv"
                    shutil.copy(
                        os.path.join(input_dir, file),
                        os.path.join(output_dir, new_name),
                    )
                    file_counters[base_name] = counter + 1


def load_neo4j_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract Neo4j configuration, preferring the second neo4j block if it exists
    neo4j_config = config.get("neo4j", {})
    if "neo4j" in config and isinstance(config["neo4j"], dict):
        neo4j_config.update(config["neo4j"])

    # Ensure all required keys are present
    required_keys = [
        "database_name",
        "wipe",
        "delimiter",
        "array_delimiter",
        "skip_duplicate_nodes",
        "skip_bad_relationships",
        "import_call_file_prefix",
        "import_call_bin_prefix",
    ]

    for key in required_keys:
        if key not in neo4j_config:
            raise ValueError(f"Missing required Neo4j configuration key: {key}")

    return neo4j_config


def load_schema_info(schema_file):
    with open(schema_file, "r") as f:
        schema = yaml.safe_load(f)
    return schema


def generate_neo4j_import_script(output_dir, neo4j_config, schema):
    output_dir_name = os.path.basename(output_dir)
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
        f'    --nodes="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/Schema_info-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/Schema_info-part.*" \\',
    ]

    # Get actual file names from the directory
    file_names = os.listdir(output_dir)

    # Separate nodes and relationships based on schema
    nodes = []
    relationships = []
    for entity, info in schema.items():
        if isinstance(info, dict):
            file_name = entity.replace(" ", "")
            if info.get("is_relationship", False):
                relationships.append(file_name)
            elif info.get("represented_as") == "node":
                nodes.append(file_name)

    # Ensure "Mentions" is in the relationships list if it exists in the directory
    if any(file.startswith("Mentions-") for file in file_names):
        relationships.append("Mentions")

    # Function to find the correct case-sensitive file name
    def find_file(base_name):
        return next(
            (f for f in file_names if f.lower() == f"{base_name.lower()}-header.csv"),
            None,
        )

    # Add nodes
    for base_name in nodes:
        file_name = find_file(base_name)
        if file_name:
            base_name = file_name[:-11]  # Remove "-header.csv"
            script_content.append(
                f'    --nodes="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part.*" \\'
            )

    # Add relationships
    for base_name in relationships:
        file_name = find_file(base_name)
        if file_name:
            base_name = file_name[:-11]  # Remove "-header.csv"
            script_content.append(
                f'    --relationships="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part.*" \\'
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
            f'    --nodes="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/Schema_info-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/Schema_info-part.*" \\',
        ]
    )

    # Add nodes for Neo4j 4.x
    for base_name in nodes:
        file_name = find_file(base_name)
        if file_name:
            base_name = file_name[:-11]  # Remove "-header.csv"
            script_content.append(
                f'    --nodes="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part.*" \\'
            )

    # Add relationships for Neo4j 4.x
    for base_name in relationships:
        file_name = find_file(base_name)
        if file_name:
            base_name = file_name[:-11]  # Remove "-header.csv"
            script_content.append(
                f'    --relationships="{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-header.csv,{neo4j_config["import_call_file_prefix"]}biocypher-out/{output_dir_name}/{base_name}-part.*" \\'
            )

    # Add database name for Neo4j 4.x
    script_content[-1] = (
        script_content[-1] + f'--database={neo4j_config["database_name"]}'
    )

    # Close the if-else statement
    script_content.append("fi")

    with open(os.path.join(output_dir, "neo4j-admin-import-call.sh"), "w") as f:
        f.write("\n".join(script_content))


def main(input_dirs, output_base_dir, config_yaml):
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

    # Remove duplicates from combined YAML
    combined_yaml = remove_duplicates(combined_yaml)

    # Write combined YAML
    with open(os.path.join(output_dir, "schema_info.yaml"), "w") as f:
        yaml.dump(combined_yaml, f)

    # Create new Schema_info CSV files
    create_schema_info_csv(output_dir, combined_yaml)

    # Verify Schema_info CSV files
    schema_header = os.path.join(output_dir, "Schema_info-header.csv")
    schema_part = os.path.join(output_dir, "Schema_info-part000.csv")
    if not (os.path.exists(schema_header) and os.path.exists(schema_part)):
        print("Error: Schema_info CSV files were not created successfully.")
        return

    # Combine CSV files (excluding Schema_info CSVs)
    combine_csv_files(input_dirs, output_dir)

    # Load Neo4j configuration
    try:
        neo4j_config = load_neo4j_config(config_yaml)
    except ValueError as e:
        print(f"Error loading Neo4j configuration: {e}")
        return

    # Generate Neo4j import script
    generate_neo4j_import_script(output_dir, neo4j_config, combined_yaml)

    # Verify the generated import script
    import_script = os.path.join(output_dir, "neo4j-admin-import-call.sh")
    if not os.path.exists(import_script):
        print("Error: Neo4j import script was not generated successfully.")
        return

    # Check if Schema_info is included in the import script
    with open(import_script, "r") as f:
        script_content = f.read()
        if "Schema_info-header.csv" not in script_content:
            print(
                "Warning: Schema_info node might not be included in the import script."
            )


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
        "--config_yaml", required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    main(args.input_dirs, args.output_base_dir, args.config_yaml)

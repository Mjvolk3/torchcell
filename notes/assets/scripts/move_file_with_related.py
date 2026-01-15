import os
import os.path as osp
import subprocess
import sys
from dotenv import load_dotenv
from os.path import splitext

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR")
VSCODE_PATH = os.environ.get("VSCODE_PATH")
PYTHON_PKG_TEST_REL_PATH = os.environ.get("PYTHON_PKG_TEST_REL_PATH")
PYTHON_PKG_REL_PATH = os.getenv("PYTHON_PKG_REL_PATH", "torchcell")

def convert_to_dendron_path(file_path):
    """Convert a file path to Dendron's period-delimited format."""
    relative_path = osp.relpath(file_path, WORKSPACE_DIR)
    file_extension = splitext(relative_path)[-1]
    # Remove extension for dendron path
    if file_extension in [".py", ".sh"]:
        dendron_path = relative_path.replace(file_extension, "").replace(osp.sep, ".")
    else:
        dendron_path = relative_path.replace(osp.sep, ".")
    return dendron_path

def handle_file(file_path, new_file_path):
    # Get file extension
    file_extension = splitext(file_path)[-1]

    # Create directories for the new file if they don't exist
    new_dir = osp.dirname(new_file_path)
    os.makedirs(new_dir, exist_ok=True)

    # Only handle test files for Python files in the library directory
    test_file_path = None
    new_test_file_path = None
    if file_extension == ".py" and file_path.startswith(osp.join(WORKSPACE_DIR, PYTHON_PKG_REL_PATH)):
        # Calculate the relative paths for the source and test files within their respective directories
        source_relative_path = osp.relpath(file_path, osp.join(WORKSPACE_DIR, PYTHON_PKG_REL_PATH))
        new_source_relative_path = osp.relpath(new_file_path, osp.join(WORKSPACE_DIR, PYTHON_PKG_REL_PATH))
        print("======================")
        print(source_relative_path)
        print(new_source_relative_path)
        # Construct the original and new test file paths
        test_file_name = "test_" + source_relative_path.split("/")[-1]
        test_file_path = osp.join(PYTHON_PKG_TEST_REL_PATH, *source_relative_path.split("/")[:-1], test_file_name)
        print("======================")
        print(test_file_name)
        print(test_file_path)

        new_test_file_name = "test_" + new_source_relative_path.split("/")[-1]
        new_test_file_path = osp.join(PYTHON_PKG_TEST_REL_PATH, *new_source_relative_path.split("/")[:-1], new_test_file_name)
        print("======================")
        print(new_test_file_name)
        print(new_test_file_path)

        if osp.exists(test_file_path):
            os.makedirs(osp.dirname(new_test_file_path), exist_ok=True)
            os.rename(test_file_path, new_test_file_path)

    # Move the file
    os.rename(file_path, new_file_path)

    # Handle Dendron notes
    dendron_old_path = convert_to_dendron_path(file_path) + ".md"
    dendron_new_path = convert_to_dendron_path(new_file_path) + ".md"

    dendron_old_full_path = osp.join(WORKSPACE_DIR, "notes", dendron_old_path)
    dendron_new_full_path = osp.join(WORKSPACE_DIR, "notes", dendron_new_path)

    if osp.exists(dendron_old_full_path):
        os.rename(dendron_old_full_path, dendron_new_full_path)

    # Open the new file, test file (if exists), and Dendron note in VS Code
    subprocess.run([VSCODE_PATH, new_file_path])
    if new_test_file_path and osp.exists(new_test_file_path):
        subprocess.run([VSCODE_PATH, new_test_file_path])
    if osp.exists(dendron_new_full_path):
        subprocess.run([VSCODE_PATH, dendron_new_full_path])

if __name__ == "__main__":
    file_path = sys.argv[1]
    new_file_path = sys.argv[2] if len(sys.argv) > 2 else ""

    # If new_file_path is empty or an unresolved VS Code variable, prompt with current as default
    if not new_file_path or new_file_path.startswith("${") or new_file_path == "${relativeFile}":
        # Get relative path from workspace root for cleaner display
        relative_path = osp.relpath(file_path, WORKSPACE_DIR)

        print(f"\nCurrent file: {relative_path}")
        print(f"Enter new path below (relative to workspace root):")
        print(f"Default: {relative_path}")
        print("-" * 50)

        user_input = input(f"New path [{relative_path}]: ").strip()

        # If empty, use the current path (no-op)
        if not user_input:
            new_file_path = file_path
        # If user input is relative (no leading /), make it relative to workspace
        elif not user_input.startswith("/"):
            new_file_path = osp.join(WORKSPACE_DIR, user_input)
        else:
            new_file_path = user_input

    # Don't proceed if paths are identical
    if file_path == new_file_path:
        print(f"Source and destination paths are the same: {file_path}")
        print("No action taken.")
        sys.exit(0)

    handle_file(file_path, new_file_path)

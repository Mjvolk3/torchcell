import os
import sys
import subprocess
import time
from dotenv import load_dotenv
from os.path import splitext

load_dotenv()

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR")
VSCODE_PATH = os.environ.get("VSCODE_PATH")
PYTHON_PKG_REL_PATH = os.getenv("PYTHON_PKG_REL_PATH", "torchcell")
PYTHON_PKG_TEST_REL_PATH = os.getenv("PYTHON_PKG_TEST_REL_PATH", "tests/torchcell")


def convert_to_dendron_path(file_path):
    """Convert a file path to Dendron's period-delimited format."""
    # Remove the leading path to the workspace
    relative_path = file_path.replace(WORKSPACE_DIR, "").lstrip("/").lstrip("\\")
    file_extension = splitext(relative_path)[-1]
    # Remove extension for dendron path
    if file_extension in [".py", ".sh"]:
        dendron_path = relative_path.replace(file_extension, "").replace("/", ".")
    else:
        dendron_path = relative_path.replace("/", ".")
    return dendron_path


def open_related_files(src_file_path):
    # Get the dendron path for the related markdown file
    print("src_file_path ", src_file_path)
    dendron_path = convert_to_dendron_path(src_file_path)
    print("dendron_path: ", dendron_path)
    md_file_path = os.path.join(WORKSPACE_DIR, "notes", dendron_path + ".md")
    print(md_file_path)

    # Check if the markdown file exists
    print(f"md_file_path: {md_file_path}")
    if os.path.exists(md_file_path):
        print("md exists")
        subprocess.run([VSCODE_PATH, md_file_path])  # Open the markdown file in VSCode

    # Only look for test file if it's a Python file in the library directory
    file_extension = splitext(src_file_path)[-1]
    if file_extension == ".py" and src_file_path.startswith(os.path.join(WORKSPACE_DIR, PYTHON_PKG_REL_PATH)):
        # Construct the test file path
        source_relative_path = os.path.relpath(src_file_path, os.path.join(WORKSPACE_DIR, PYTHON_PKG_REL_PATH))
        test_file_name = "test_" + os.path.basename(src_file_path)
        test_file_path = os.path.join(WORKSPACE_DIR, PYTHON_PKG_TEST_REL_PATH, *source_relative_path.split("/")[:-1], test_file_name)
        print(f"test_file_path: {test_file_path}")

        # Check if the test file exists
        if os.path.exists(test_file_path):
            print("test exists")
            subprocess.run([VSCODE_PATH, test_file_path])  # Open the test file in VSCode

    # Add a delay to ensure files are opened before the prompt
    time.sleep(2)

    # Ask for user confirmation to delete the source file
    confirm = input(f"Do you want to delete the source file {src_file_path}? (yes/no): ")
    if confirm.lower() == "yes":
        if os.path.exists(src_file_path):
            os.remove(src_file_path)
        print("Src File deleted.")
    else:
        print("Deletion cancelled.")


if __name__ == "__main__":
    src_file_path = sys.argv[1]
    open_related_files(src_file_path)

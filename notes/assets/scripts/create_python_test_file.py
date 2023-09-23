import os
import sys
import subprocess
from dotenv import load_dotenv
from pathlib import Path

# Load the environment variables
load_dotenv()

# Get the environment variables
PYTHON_PKG_REL_PATH = os.getenv("PYTHON_PKG_REL_PATH")
PYTHON_PKG_TEST_REL_PATH = os.getenv("PYTHON_PKG_TEST_REL_PATH")
VSCODE_PATH = os.getenv("VSCODE_PATH")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")


def create_test_file(src_file_path):
    print('=' * 80)
    # Get the relative path from the source directory to WORKSPACE_DIR
    src_directory = os.path.dirname(src_file_path)
    rel_path_from_workspace_to_src_dir = os.path.relpath(src_directory, WORKSPACE_DIR)

    print(
        f"Relative path from workspace to src directory: {rel_path_from_workspace_to_src_dir}"
    )

    # Form new path replacing src with tests
    test_directory = (
        Path(rel_path_from_workspace_to_src_dir).as_posix().replace("src", "tests")
    )
    test_directory = os.path.join(WORKSPACE_DIR, test_directory)

    print(f"Test directory after forming new path: {test_directory}")

    # Extract the base filename, prepend "test_", and then reconstruct the path
    base_filename = os.path.basename(src_file_path)
    test_filename = "test_" + base_filename

    # Construct the full path of the test file
    test_file_path = os.path.join(test_directory, test_filename)

    # If the test file doesn't exist, create it
    if not os.path.exists(test_file_path):
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        with open(test_file_path, "w") as test_file:
            test_file.write("# Test file\n")
        print(f"Test file created at: {test_file_path}")
        subprocess.run(
            [VSCODE_PATH, test_file_path]
        )  # Open the created test file in VSCode
    else:
        print(f"Test file already exists at: {test_file_path}")
        subprocess.run(
            [VSCODE_PATH, test_file_path]
        )  # Open the existing test file in VSCode
    print('=' * 80)

if __name__ == "__main__":
    event = sys.argv[1]
    src_file_path = sys.argv[2]

    if event == "create":
        create_test_file(src_file_path)

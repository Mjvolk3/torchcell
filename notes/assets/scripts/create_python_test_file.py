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


def get_git_root():
    """Get the git repository root, works in both main repo and worktrees."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # Fallback to get_git_root() if not in a git repo
        return os.environ.get("get_git_root()")


def create_test_file(src_file_path):
    print('=' * 80)

    # Only work on files in the library directory (torchcell/)
    if not src_file_path.startswith(os.path.join(get_git_root(), PYTHON_PKG_REL_PATH)):
        print(f"Error: Test files can only be created for Python files in the {PYTHON_PKG_REL_PATH}/ directory.")
        print(f"File provided: {src_file_path}")
        print(f"Expected to start with: {os.path.join(get_git_root(), PYTHON_PKG_REL_PATH)}")
        print("Test file creation cancelled.")
        print('=' * 80)
        return

    # Get the relative path from the source directory to get_git_root()
    rel_path_from_workspace_to_src_file = os.path.relpath(src_file_path, get_git_root())

    print(
        f"Relative path from workspace to source file: {rel_path_from_workspace_to_src_file}"
    )

    # Form new path using PYTHON_PKG_TEST_REL_PATH
    test_file_relative_path = rel_path_from_workspace_to_src_file.replace(PYTHON_PKG_REL_PATH, PYTHON_PKG_TEST_REL_PATH)

    # Construct the full path of the test file
    test_file_path = os.path.join(get_git_root(), test_file_relative_path)
    test_file_directory = os.path.dirname(test_file_path)
    test_filename = "test_" + os.path.basename(src_file_path)
    test_file_path = os.path.join(test_file_directory, test_filename)

    print(f"Test file path: {test_file_path}")

    # If the test file doesn't exist, create it
    if not os.path.exists(test_file_path):
        os.makedirs(test_file_directory, exist_ok=True)
        with open(test_file_path, "w") as test_file:
            test_file.write("# Test file\n")
        print(f"Test file created at: {test_file_path}")
    else:
        print(f"Test file already exists at: {test_file_path}")

    # Open the test file in VSCode
    subprocess.run([VSCODE_PATH, test_file_path])
    print('=' * 80)

if __name__ == "__main__":
    event = sys.argv[1]
    src_file_path = sys.argv[2]

    if event == "create":
        create_test_file(src_file_path)

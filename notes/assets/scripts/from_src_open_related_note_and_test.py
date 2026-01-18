import os
import sys
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

VSCODE_PATH = os.environ.get("VSCODE_PATH")


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


def convert_to_dendron_path(file_path):
    """Convert a file path to Dendron's period-delimited format."""
    git_root = get_git_root()
    # Remove the leading path to the workspace
    relative_path = file_path.replace(git_root, "")
    return relative_path.replace("/", ".").replace(".py", "")


def open_related_files(src_file_path):
    # Get the dendron path for the related markdown file
    print("src_file_path ", src_file_path)
    dendron_path = convert_to_dendron_path(src_file_path)[1:]
    print("dendron_path: ", dendron_path)
    md_file_path = os.path.join(
        get_git_root(), "notes", dendron_path + ".md"
    )
    print(md_file_path)

    # Construct the test file path
    dir_path, filename = os.path.split(src_file_path)
    test_filename = "test_" + filename
    test_file_path = os.path.join(dir_path.replace("src", "tests"), test_filename)
    print(test_file_path)
    # Check if the markdown file exists
    print(f"md_file_path: {md_file_path}")
    if os.path.exists(md_file_path):
        print("md exists")
        subprocess.run(
            [VSCODE_PATH, md_file_path]
        )  # Open the markdown file in VSCode

    # Check if the test file exists
    print(f"test_file_path: {test_file_path}")
    if os.path.exists(test_file_path):
        print("test exists")
        subprocess.run(
            [VSCODE_PATH, test_file_path]
        )  # Open the test file in VSCode

if __name__ == "__main__":
    src_file_path = sys.argv[1]
    open_related_files(src_file_path)

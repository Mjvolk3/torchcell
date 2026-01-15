# /Users/michaelvolk/Documents/projects/torchcell/notes/assets/scripts/from_note_open_related_file.py
import os
import sys
import subprocess
from dotenv import load_dotenv

load_dotenv()

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR")
VSCODE_PATH = os.environ.get("VSCODE_PATH")


def convert_to_file_path(dendron_path, extension):
    """Convert Dendron's period-delimited format to a file path with given extension."""
    # Append the leading path to the workspace
    return os.path.join(WORKSPACE_DIR, dendron_path.replace(".", "/") + extension)


def open_related_file(note_file_path):
    print("note_file_path ", note_file_path)

    # Extract the dendron path from the note file path
    dendron_path = (
        note_file_path.replace(WORKSPACE_DIR, "")
        .replace("notes", "")
        .replace(".md", "")
        .lstrip("/")
        .lstrip("\\")
    )
    print("dendron_path: ", dendron_path)

    # Try both .py and .sh extensions
    for extension in [".py", ".sh"]:
        file_path = convert_to_file_path(dendron_path, extension)
        print(f"Checking {extension} file: {file_path}")
        if os.path.exists(file_path):
            print(f"{extension} file exists")
            subprocess.run([VSCODE_PATH, file_path])  # Open the related file in VSCode
            return

    print("No related .py or .sh file found")


if __name__ == "__main__":
    note_file_path = sys.argv[1]
    open_related_file(note_file_path)

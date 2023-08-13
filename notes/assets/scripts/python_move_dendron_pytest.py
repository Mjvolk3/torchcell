import os
import os.path as osp
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR")
VSCODE_PATH = os.environ.get("VSCODE_PATH")


def convert_to_dendron_path(file_path):
    """Convert a file path to Dendron's period-delimited format."""
    relative_path = file_path.replace(WORKSPACE_DIR, "")
    dendron_path = relative_path.replace("/", ".").replace(".py", "")
    return dendron_path


def handle_python_file(file_path, new_file_path):
    base_dir = "notes/"
    ws_root = WORKSPACE_DIR

    dendron_new_path = convert_to_dendron_path(new_file_path)
    dendron_old_path = convert_to_dendron_path(file_path)[1:]

    dendron_note_exists = osp.exists(
        osp.join(ws_root, base_dir, dendron_old_path + ".md")
    )

    new_dir = osp.dirname(new_file_path)
    if not osp.exists(new_dir):
        os.makedirs(new_dir)

    original_filename = osp.basename(file_path)
    test_file_path = file_path.replace("src/torchcell", "tests/torchcell").replace(
        original_filename, "test_" + original_filename
    )
    test_new_file_path = new_file_path.replace(
        "src/torchcell", "tests/torchcell"
    ).replace(original_filename, "test_" + original_filename)

    # Check if destination test or markdown file exists
    if osp.exists(test_new_file_path):
        print(
            f"Destination test file {test_new_file_path} already exists. Opening it in VSCode."
        )
        subprocess.run([VSCODE_PATH, test_new_file_path])
        return

    if osp.exists(
        osp.join(ws_root, base_dir, dendron_new_path.replace("..", ".") + ".md")
    ):
        print(
            f"Destination markdown file {dendron_new_path.replace('..', '.')} already exists. Opening it in VSCode."
        )
        subprocess.run(
            [
                VSCODE_PATH,
                osp.join(
                    ws_root, base_dir, dendron_new_path.replace("..", ".") + ".md"
                ),
            ]
        )
        return

    os.rename(file_path, new_file_path)
    subprocess.run([VSCODE_PATH, new_file_path])

    # If the Dendron note exists, move it
    if dendron_note_exists:
        cmd = [
            "dendron",
            "note",
            "move",
            "--wsRoot",
            ws_root,
            "--vault",
            "torchcell",
            "--fname",
            dendron_old_path,
            "--destFname",
            dendron_new_path,
        ]
        subprocess.run(cmd, check=True)
        dendron_new_file_path = osp.join(
            ws_root, base_dir, dendron_new_path.replace("..", ".") + ".md"
        )
        subprocess.run([VSCODE_PATH, dendron_new_file_path])

    if osp.exists(test_file_path):
        test_new_dir = osp.dirname(test_new_file_path)
        if not osp.exists(test_new_dir):
            os.makedirs(test_new_dir)

        os.rename(test_file_path, test_new_file_path)
        subprocess.run([VSCODE_PATH, test_new_file_path])
    else:
        print(f"Test file {test_file_path} does not exist. Skipping.")


if __name__ == "__main__":
    file_path = sys.argv[1]
    new_file_path = sys.argv[2]

    handle_python_file(file_path, new_file_path)

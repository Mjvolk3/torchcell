import os
import sys
import subprocess
from dotenv import load_dotenv
load_dotenv()

PYTHON_PKG_REL_PATH = os.environ["PYTHON_PKG_REL_PATH"]
PYTHON_PKG_TEST_REL_PATH = os.environ["PYTHON_PKG_TEST_REL_PATH"]
VSCODE_PATH = os.environ.get("VSCODE_PATH")

def create_test_file(src_file_path):
    print(f"Received source file path: {src_file_path}")

    # Extract the base filename, prepend "test_", and then reconstruct the path
    base_filename = os.path.basename(src_file_path)
    test_filename = "test_" + base_filename
    test_file_path = os.path.join(
        os.path.dirname(src_file_path).replace(
            PYTHON_PKG_REL_PATH, PYTHON_PKG_TEST_REL_PATH
        ),
        os.path.dirname(src_file_path).replace(PYTHON_PKG_REL_PATH, PYTHON_PKG_TEST_REL_PATH),
        test_filename,
    )
    print(f"Generated test file path: {test_file_path}")

    # If the test file doesn't exist, create it
    if not os.path.exists(test_file_path):
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        with open(test_file_path, "w") as test_file:
            test_file.write("# Test file\n")
        print(f"Test file created at: {test_file_path}")
        subprocess.run([VSCODE_PATH, test_file_path])  # Open the created test file in VSCode
    else:
        print(f"Test file already exists at: {test_file_path}")
        subprocess.run([VSCODE_PATH, test_file_path])  # Open the created test file in VSCode

if __name__ == "__main__":
    event = sys.argv[1]
    src_file_path = sys.argv[2]

    if event == "create":
        create_test_file(src_file_path)

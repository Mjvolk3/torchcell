# notes/assets/scripts/add_frontmatter
# [[notes.assets.scripts.add_frontmatter]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/scripts/add_frontmatter

import os
import os.path as osp
import sys
from dotenv import load_dotenv
load_dotenv()
from os.path import splitext
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR")
PYTHON_PKG_REL_PATH = os.environ.get("PYTHON_PKG_REL_PATH")
PYTHON_PKG_TEST_REL_PATH = os.environ.get("PYTHON_PKG_TEST_REL_PATH")
GIT_REPO_URL = os.environ.get("GIT_REPO_URL")

def add_frontmatter(file_path):
    # Extract the relative path
    print(f"file path:{file_path}")
    relative_path = osp.relpath(
        file_path, start=WORKSPACE_DIR
    )
    print(f"relative path:{relative_path}")

    # Get file extension
    file_extension = splitext(relative_path)[-1]

    # Remove extension from relative_path for note naming
    if file_extension in [".py", ".sh"]:
        relative_path_no_ext = relative_path.replace(file_extension, '')
    else:
        relative_path_no_ext = relative_path

    # Determine if we should include test file line
    # Only for .py files in the library directory (torchcell/)
    include_test_file = False
    test_file_path = None
    if file_extension == ".py" and relative_path.startswith(PYTHON_PKG_REL_PATH):
        include_test_file = True
        # Generate the test file path
        test_file_path = relative_path.replace(PYTHON_PKG_REL_PATH, PYTHON_PKG_TEST_REL_PATH)
        test_file_path = osp.join(
            osp.dirname(test_file_path), "test_" + osp.basename(test_file_path)
        )

    # Generate the frontmatter lines
    lines = [
        f"# {relative_path_no_ext}\n",
        f"# [[{relative_path_no_ext.replace('/', '.')}]]\n",
        f"# {GIT_REPO_URL}/tree/main/{relative_path_no_ext}\n",
    ]

    if include_test_file:
        lines.append(f"# Test file: {test_file_path}\n")

    lines.append("\n")  # Add an extra newline for separation

    with open(file_path, "r+") as file:
        content = file.readlines()

        print(
            f"Debug: First line of the file: {content[0] if content else 'File is empty'}"
        )

        # Check if file has a shebang (works for both .py and .sh files)
        has_shebang = content and content[0].startswith("#!")

        if has_shebang:
            # Check if frontmatter exists after shebang
            if len(content) > 1 and content[1].startswith("# " + relative_path_no_ext):
                print("Frontmatter already exists after shebang.")
                return
            # Insert frontmatter after shebang
            shebang = content[0]
            content = [shebang] + lines + content[1:]
        else:
            # No shebang, check if frontmatter exists at beginning
            if content and content[0].startswith("# " + relative_path_no_ext):
                print("Frontmatter already exists.")
                return
            # Add frontmatter at beginning
            content = lines + content

        file.seek(0)
        file.writelines(content)
        file.truncate()  # Ensure any leftover content is removed

    print("Frontmatter added successfully.")


if __name__ == "__main__":
    file_path = sys.argv[1]
    add_frontmatter(file_path)

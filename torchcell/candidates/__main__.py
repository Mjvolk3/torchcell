# torchcell/candidates/__main__.py
# [[torchcell.candidates.__main__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/candidates/__main__.py
"""``python -m torchcell.candidates``: the ``candidate-gate`` CLI."""

import sys

from torchcell.candidates.cli import main

if __name__ == "__main__":
    sys.exit(main())

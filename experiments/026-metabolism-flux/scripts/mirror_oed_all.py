# experiments/026-metabolism-flux/scripts/mirror_oed_all.py
# [[experiments.026-metabolism-flux.scripts.mirror_oed_all]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/mirror_oed_all.py

"""Mirror the WHOLE Open Enzyme Database, every organism, beside the yeast slice.

The yeast slice (1,126 records, 90 EC4 classes, four of them with three or more distinct
enzymes) cannot say how much enzymes of the same function vary, which is the question an
EC-class variance analysis asks. The full database can: 35,753 records at retrieval.

Same provenance discipline as the yeast slice: the stored bytes plus their sha256 are
canonical, the URL is retrieval metadata, and ``load_mirrored_records`` refuses a file
whose hash does not match its manifest.

Writes ``$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/all/{oed_records,manifest}.json``.
"""

import os
import os.path as osp
from typing import cast

from dotenv import load_dotenv

from torchcell.metabolism.enzyme_kinetics import mirror_oed_slice

load_dotenv()
DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
MIRROR_DIR = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "open_enzyme_database", "all")


def main() -> None:
    record = mirror_oed_slice(MIRROR_DIR, organism=None, page_size=1000)
    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

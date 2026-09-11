"""Create the Neo4j database directory tree and copy in config files.

``--env-file`` names the container ``.env`` to copy in (default
``$WORKSPACE_DIR/database/database.env``). It is a machine-local secrets file that is not
tracked, so a git worktree does not have one; a build driven from a worktree passes the
primary checkout's copy explicitly.
"""

import argparse
import os
import os.path as osp
import shutil
from typing import cast

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
WORKSPACE_DIR = cast(str, os.getenv("WORKSPACE_DIR"))


def main(argv: list[str] | None = None) -> None:
    """Build the database directory layout and copy conf, env, and biocypher files."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.database.directory_setup",
        description="Create $DATA_ROOT/database/... and copy conf, env, biocypher in.",
    )
    parser.add_argument(
        "--env-file",
        default=osp.join(WORKSPACE_DIR, "database", "database.env"),
        help="container .env to copy to $DATA_ROOT/database/.env",
    )
    args = parser.parse_args(argv)
    # Create directories
    directories = [
        osp.join(DATA_ROOT, "database"),
        osp.join(DATA_ROOT, "database/data/torchcell"),
        osp.join(DATA_ROOT, "database/data"),
        osp.join(DATA_ROOT, "database/biocypher"),
        osp.join(DATA_ROOT, "database/conf"),
        osp.join(DATA_ROOT, "database/logs"),
        osp.join(DATA_ROOT, "database/slurm"),
        osp.join(DATA_ROOT, "database/plugins"),
        osp.join(DATA_ROOT, "database/metrics"),
        osp.join(DATA_ROOT, "database/import"),
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

    # Copy neo4j.conf
    src_neo4j_conf = osp.join(WORKSPACE_DIR, "database", "conf", "gh_neo4j.conf")
    dst_neo4j_conf = osp.join(DATA_ROOT, "database/conf/neo4j.conf")
    shutil.copyfile(src_neo4j_conf, dst_neo4j_conf)

    # Copy and rename .env file
    dst_env_path = osp.join(DATA_ROOT, "database/.env")
    shutil.copyfile(args.env_file, dst_env_path)

    # Copy biocypher directory into database
    src_biocypher = osp.join(WORKSPACE_DIR, "biocypher")
    dst_biocypher = osp.join(DATA_ROOT, "database/biocypher")
    if osp.exists(dst_biocypher):
        shutil.rmtree(dst_biocypher)
    shutil.copytree(src_biocypher, dst_biocypher)

    print("Setup completed successfully.")


if __name__ == "__main__":
    main()

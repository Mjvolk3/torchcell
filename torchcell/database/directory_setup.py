from dotenv import load_dotenv
import os
import os.path as osp
import shutil

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")

def main() -> None:
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
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

    # Copy neo4j.conf
    src_neo4j_conf = osp.join(WORKSPACE_DIR, "database", "conf", "gh_neo4j.conf")
    dst_neo4j_conf = osp.join(DATA_ROOT, "database/conf/neo4j.conf")
    shutil.copyfile(src_neo4j_conf, dst_neo4j_conf)

    # Copy and rename .env file
    src_database_env = osp.join(WORKSPACE_DIR, "database", "database.env")
    dst_env_path = osp.join(DATA_ROOT, "database/.env")
    shutil.copyfile(src_database_env, dst_env_path)

    # Copy biocypher directory into database
    src_biocypher = osp.join(WORKSPACE_DIR, "biocypher")
    dst_biocypher = osp.join(DATA_ROOT, "database/biocypher")
    if osp.exists(dst_biocypher):
        shutil.rmtree(dst_biocypher)
    shutil.copytree(src_biocypher, dst_biocypher)

    print("Setup completed successfully.")

if __name__ == "__main__":
    main()
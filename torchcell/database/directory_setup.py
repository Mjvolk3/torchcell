from dotenv import load_dotenv
import os
import os.path as osp
import shutil

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")

def main() -> None:
    # create database dir in DATA_ROOT and make dir
    database_dir = osp.join(DATA_ROOT, "database")
    os.makedirs(database_dir, exist_ok=True)
    tc_data_dir = osp.join(DATA_ROOT, "data/torchcell")
    os.makedirs(tc_data_dir, exist_ok=True)

    # make the following directories in database_dir: data, biocypher, conf, logs, and slurm
    data_dir = osp.join(database_dir, "data")
    biocypher_dir = osp.join(database_dir, "biocypher")
    conf_dir = osp.join(database_dir, "conf")
    logs_dir = osp.join(database_dir, "logs")
    slurm_dir = osp.join(database_dir, "slurm")
    
    for dir_path in [data_dir, biocypher_dir, conf_dir, logs_dir, slurm_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # copy osp.join(WORKSPACE_DIR, "database/conf/gh_neo4j.conf") to osp.join(DATA_ROOT, "database/conf") and rename it to "neo4j.conf"
    src_neo4j_conf = osp.join(WORKSPACE_DIR, "database", "conf", "gh_neo4j.conf")
    dst_neo4j_conf = osp.join(conf_dir, "neo4j.conf")
    shutil.copyfile(src_neo4j_conf, dst_neo4j_conf)

    # Ensure .env is a file, not a directory
    env_path = osp.join(database_dir, ".env")
    if osp.isdir(env_path):
        shutil.rmtree(env_path)

    # copy osp.join(WORKSPACE_DIR, "database/database.env") to osp.join(DATA_ROOT, "database") and rename it to ".env"
    src_database_env = osp.join(WORKSPACE_DIR, "database", "database.env")
    shutil.copyfile(src_database_env, env_path)

    # copy biocypher directory
    src_biocypher = osp.join(WORKSPACE_DIR, "biocypher")
    dst_biocypher = osp.join(DATA_ROOT, "biocypher")
    if osp.exists(dst_biocypher):
        shutil.rmtree(dst_biocypher)
    shutil.copytree(src_biocypher, dst_biocypher)

if __name__ == "__main__":
    main()
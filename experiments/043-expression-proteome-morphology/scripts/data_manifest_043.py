# experiments/043-expression-proteome-morphology/scripts/data_manifest_043.py
# [[experiments.043-expression-proteome-morphology.scripts.data_manifest_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/data_manifest_043.py
"""A per-machine manifest of the 043 training data, and a comparison of manifests.

Stdlib only and Python 3.6 compatible (the ALCF login nodes run 3.6.15), so the file can
be copied to any machine and run with the stock ``python3``:

    python3 data_manifest_043.py --root <DATA_ROOT> --out <machine>.json [--repo <tree>]
    python3 data_manifest_043.py --compare gilahyper.json polaris.json aurora.json ...

``--root`` is the directory the trainer reads as ``$DATA_ROOT`` (on ALCF that is
``<project>/mjvolk3/torchcell-data/043``). ``--repo`` is the code tree the jobs run from:
the split files under ``conf/splits/`` travel with the code, not the data, so they are
hashed there, together with ``.synced_commit`` and the git HEAD when the tree is a repo.

Tiers (what ``train_043.py`` opens, read from the code on 2026-10-28):

``train``     opened by a stage-1 or fine-tune run with ``data.prep_version=prep_v4``:
              ``PreparedStore043.load`` reads ``meta.json``, ``store.npz`` and
              ``experiment_ontology_vocab.json`` (sha256-checked against ``meta.json``) of
              ``train_043_prep/prep_v4_route{A,B}``; ``h5_vector_source`` reads
              ``representation_043_v2/vocab_<encoder>.npy`` and
              ``isolate_embeddings/peter2018_1011/alleles/<encoder>.h5`` for the four
              encoders. Nothing under ``raw_043_v2`` is opened at train time.
``train_v3``  ``prep_v3_routeA``, what a run with ``data.prep_version`` unset (null) opens.
``companion`` shipped with the train tier, not opened by it: the rest of
              ``representation_043_v2``, the allele-table manifest and verify report.
``prepare``   read only by ``--stage prepare``: ``raw_043_v2/raw`` (LMDB + indexes; the
              split's ``check_store`` hashes ``raw_dataset_name_index.json``), the
              ``raw_043_v2/processed`` metadata (not its LMDB), ``ontology_043_v2``.
              The genome / GO / STRING / TFLink roots of ``_edge_lists`` are not listed.

Hashes: files under 2 GiB get a full sha256. Larger files get the sha256 of the first and
of the last 64 MiB (``method: head_tail_64MiB``), plus the full sha256 only with
``--full`` (GilaHyper under slurm); the comparison uses whatever both manifests carry.
LMDB ``lock.mdb`` and ``*.lock`` files are listed by size only (they are runtime state).
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import socket
import subprocess
import sys
import time

EXPERIMENT = "043-expression-proteome-morphology"
EXP_REL = "data/torchcell/experiments/" + EXPERIMENT
ISO_REL = "data/scerevisiae/isolate_embeddings/peter2018_1011"
ENCODERS = (
    "fudt_species_upstream",
    "fudt_species_downstream",
    "prot_t5_xl_uniref50",
    "calm",
)
PREP_FILES = ("meta.json", "store.npz", "experiment_ontology_vocab.json")
FULL_LIMIT = 2 * 1024**3
CHUNK = 64 * 1024**2
#: Fields of meta.json kept verbatim; long lists are reduced to length + sha256.
LIST_LIMIT = 50
#: Code files hashed under --repo (split files travel with the code).
REPO_FILES = (
    "experiments/" + EXPERIMENT + "/conf/splits/global_split_v1.json",
    "experiments/" + EXPERIMENT + "/conf/splits/global_split_v2.json",
    "experiments/" + EXPERIMENT + "/conf/splits/global_split_v2_redraw.json",
    "experiments/" + EXPERIMENT + "/conf/train_043_default.yaml",
    "experiments/" + EXPERIMENT + "/scripts/train_043.py",
    "experiments/" + EXPERIMENT + "/scripts/global_split_043.py",
    "experiments/" + EXPERIMENT + "/scripts/store_version_043.py",
    "torchcell/models/cell_model_043.py",
    "torchcell/data/representation_043.py",
    "torchcell/data/experiment_ontology.py",
    "torchcell/data/target_transform_043.py",
)


def required_files():
    """``[(tier, relative path)]`` of the individually named files."""
    out = []
    for route in ("A", "B"):
        d = EXP_REL + "/train_043_prep/prep_v4_route" + route
        out += [("train", d + "/" + f) for f in PREP_FILES]
    out += [
        ("train", EXP_REL + "/representation_043_v2/vocab_" + e + ".npy")
        for e in ENCODERS
    ]
    out += [("train", ISO_REL + "/alleles/" + e + ".h5") for e in ENCODERS]
    out += [
        ("train_v3", EXP_REL + "/train_043_prep/prep_v3_routeA/" + f)
        for f in ("meta.json", "store.npz")
    ]
    for f in (
        "accessory.parquet",
        "build_record.json",
        "contexts.json",
        "contexts.npz",
        "reporter.json",
        "reporter.parquet",
        "segregant_alleles.parquet",
        "verify_records.parquet",
    ):
        out.append(("companion", EXP_REL + "/representation_043_v2/" + f))
    out += [
        ("companion", ISO_REL + "/manifest.json"),
        ("companion", ISO_REL + "/verify_report.json"),
    ]
    for f in (
        "raw/raw_dataset_name_index.json",
        "raw/experiment_reference_index.json",
        "raw/gene_set.json",
        "raw/lmdb/data.mdb",
        "processed/dataset_name_index.json",
        "processed/experiment_types.json",
        "processed/gene_set.json",
        "processed/label_df.parquet",
        "processed/perturbation_count_index.json",
        "processed/phenotype_label_index.json",
        "processed/pre_filter.pt",
        "processed/pre_transform.pt",
    ):
        out.append(("prepare", EXP_REL + "/raw_043_v2/" + f))
    out.append(("prepare", EXP_REL + "/ontology_043_v2/experiment_ontology_vocab.json"))
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024**2)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def head_tail(path, size):
    with open(path, "rb") as f:
        head = hashlib.sha256(f.read(CHUNK)).hexdigest()
        f.seek(max(size - CHUNK, 0))
        tail = hashlib.sha256(f.read(CHUNK)).hexdigest()
    return head, tail


def describe(root, rel, full):
    path = osp.join(root, rel)
    if not osp.isfile(path):
        return {"present": False}
    st = os.stat(path)
    rec = {
        "present": True,
        "size": st.st_size,
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
    }
    base = osp.basename(rel)
    if base == "lock.mdb" or base.endswith(".lock"):
        rec["method"] = "size_only_runtime_state"
        return rec
    if st.st_size < FULL_LIMIT:
        rec["method"] = "full"
        rec["sha256"] = sha256_file(path)
        return rec
    rec["method"] = "head_tail_64MiB"
    rec["head_sha256"], rec["tail_sha256"] = head_tail(path, st.st_size)
    if full:
        rec["method"] = "full+head_tail_64MiB"
        rec["sha256"] = sha256_file(path)
    return rec


def reduce_meta(meta):
    out = {}
    for k, v in meta.items():
        if isinstance(v, (list, dict)) and len(v) > LIST_LIMIT:
            blob = json.dumps(v, sort_keys=True).encode()
            out[k] = {"len": len(v), "sha256": hashlib.sha256(blob).hexdigest()}
        else:
            out[k] = v
    return out


def git_head(tree):
    try:
        r = subprocess.run(
            ["git", "-C", tree, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"head": None, "error": str(exc)}
    if r.returncode != 0:
        return {"head": None, "error": r.stderr.strip().splitlines()[-1:]}
    return {"head": r.stdout.strip()}


def repo_info(tree):
    info = {"tree": osp.abspath(tree), "git": git_head(tree), "files": {}}
    sc = osp.join(tree, ".synced_commit")
    info["synced_commit"] = open(sc).read().strip() if osp.isfile(sc) else None
    for rel in REPO_FILES:
        info["files"][rel] = describe(tree, rel, False)
    split = osp.join(
        tree, "experiments", EXPERIMENT, "conf", "splits", "global_split_v2.json"
    )
    if osp.isfile(split):
        with open(split) as f:
            s = json.load(f)
        info["global_split_v2"] = {
            "version": s.get("version"),
            "store_sha256": s.get("store_sha256"),
        }
    return info


def build(args):
    root = osp.abspath(args.root)
    files = {}
    for tier, rel in required_files():
        rec = describe(root, rel, args.full)
        rec["tier"] = tier
        files[rel] = rec
        print(f"{tier:<9} {'ok' if rec['present'] else 'MISSING':<7} {rel}")
        sys.stdout.flush()
    lock = EXP_REL + "/raw_043_v2/raw/lmdb/lock.mdb"
    files[lock] = dict(describe(root, lock, False), tier="prepare")
    prep_meta = {}
    for tag in ("prep_v3_routeA", "prep_v4_routeA", "prep_v4_routeB"):
        p = osp.join(root, EXP_REL, "train_043_prep", tag, "meta.json")
        if osp.isfile(p):
            with open(p) as f:
                prep_meta[tag] = reduce_meta(json.load(f))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "machine": args.machine,
        "root": root,
        "full_hash_limit_bytes": FULL_LIMIT,
        "head_tail_bytes": CHUNK,
        "full_flag": bool(args.full),
        "files": files,
        "prep_meta": prep_meta,
        "repo": repo_info(args.repo) if args.repo else None,
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("wrote " + args.out)


def _hash_cmp(a, b):
    """``None`` = equal, else a description; compares what both records carry."""
    if a["size"] != b["size"]:
        return f"size {a['size']} != {b['size']}"
    if "sha256" in a and "sha256" in b:
        if a["sha256"] != b["sha256"]:
            return "sha256 {} != {}".format(a["sha256"], b["sha256"])
        return None
    if "head_sha256" in a and "head_sha256" in b:
        for k in ("head_sha256", "tail_sha256"):
            if a[k] != b[k]:
                return f"{k} {a[k]} != {b[k]}"
        return None
    return "not comparable ({} vs {})".format(a.get("method"), b.get("method"))


def compare(paths):
    ms = []
    for p in paths:
        with open(p) as f:
            ms.append(json.load(f))
    ref = ms[0]
    name = lambda m: m.get("machine") or m["host"]  # noqa: E731
    print(f"reference: {name(ref)} ({paths[0]})")
    n_issue = 0
    for m in ms[1:]:
        print(f"\n== {name(m)} vs {name(ref)}")
        for rel in sorted(ref["files"]):
            a = ref["files"][rel]
            b = m["files"].get(rel, {"present": False})
            tier = a["tier"]
            if not a["present"] and not b["present"]:
                continue
            if a["present"] != b["present"]:
                n_issue += 1
                where = name(m) if a["present"] else name(ref)
                print(f"  [{tier}] MISSING on {where}: {rel}")
                continue
            if a.get("method") == "size_only_runtime_state":
                continue
            d = _hash_cmp(a, b)
            if d is not None:
                n_issue += 1
                print(f"  [{tier}] DIFFERS {rel}: {d}")
        for tag in sorted(set(ref["prep_meta"]) | set(m["prep_meta"])):
            ma, mb = ref["prep_meta"].get(tag), m["prep_meta"].get(tag)
            if ma is None or mb is None:
                continue
            for k in sorted(set(ma) | set(mb)):
                if ma.get(k) != mb.get(k):
                    print(
                        f"  meta {tag}.{k}: {json.dumps(mb.get(k))} != {json.dumps(ma.get(k))}"
                    )
        ra, rb = ref.get("repo"), m.get("repo")
        if ra and rb:
            print(
                "  code: synced_commit {} git {} (ref synced {} git {})".format(
                    rb["synced_commit"],
                    rb["git"].get("head"),
                    ra["synced_commit"],
                    ra["git"].get("head"),
                )
            )
            for rel in sorted(ra["files"]):
                fa, fb = ra["files"][rel], rb["files"].get(rel, {"present": False})
                if fa["present"] != fb["present"]:
                    print(
                        "  code MISSING on {}: {}".format(
                            name(m) if fa["present"] else name(ref), rel
                        )
                    )
                elif fa["present"] and fa["sha256"] != fb["sha256"]:
                    print(f"  code DIFFERS {rel}")
    print(f"\n{n_issue} data issue(s)")
    return n_issue


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="the DATA_ROOT the trainer reads")
    ap.add_argument("--out", help="manifest JSON to write")
    ap.add_argument("--repo", default=None, help="code tree the jobs run from")
    ap.add_argument("--machine", default=None, help="label, e.g. gilahyper")
    ap.add_argument("--full", action="store_true", help="full sha256 of files >2 GiB")
    ap.add_argument("--compare", nargs="+", default=None, help="reference first")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        if not (args.root and args.out):
            ap.error("--root and --out are required unless --compare")
        build(args)


if __name__ == "__main__":
    main()

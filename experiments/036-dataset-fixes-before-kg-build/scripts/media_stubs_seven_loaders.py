# experiments/036-dataset-fixes-before-kg-build/scripts/media_stubs_seven_loaders.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.media_stubs_seven_loaders]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/media_stubs_seven_loaders
"""Issue #622 before/after: the media seven loaders emit, stub versus sourced object.

For each of the seven loaders it takes the environment the loader now emits (the
module-level medium object its ``create_experiment`` uses, or Nadal-Ribelles' static
``_environment`` builder for both conditions) and records the medium's name, state,
``is_synthetic``, ``base_medium``, component count, provenance count, open-gap count,
whether its ``media_identity`` equals a library object's, the environment's temperature
and gaps, and, for every SourcedValue the loader added, the mirror file line the quote
sits on (found by searching the pinned file, whose sha256 is re-checked).

It then reads each loader's DEV store (``$DATA_ROOT/data/torchcell/<slug>/processed``)
and tallies the distinct media there (the "before"). With ``--build`` it builds the four
small datasets (Ozaydin 2013, Yoshida 2012, da Silveira 2014, SynLethDB SL and SR) into
``--scratch-root`` and tallies their media too (the "after"), plus the record count,
and checks that every scratch record's medium equals the loader object. Caudal 2024,
Ohya 2005, Ohnuki 2018 and Nadal-Ribelles 2025 are large and are NOT built here; their
dev rebuild runs under slurm.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/media_stubs_seven_loaders.csv``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/media_stubs_seven_loaders.py \
        --scratch-root <dir> [--build]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import pickle
import shutil
from collections import Counter
from typing import Any

import lmdb
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datamodels.identity import media_identity
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import Environment, Media
from torchcell.datasets.scerevisiae import (
    caudal2024,
    dasilveira2014,
    nadal_ribelles2025,
    ohnuki2018,
    ohya2005,
    ozaydin2013,
    synth_leth_db,
    yoshida2012,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.sourced import SourcedValue

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "media_stubs_seven_loaders.csv",
)


def _nadal_environments() -> list[Environment | Media]:
    """Both Nadal-Ribelles conditions, through the loader's own builder."""
    build = nadal_ribelles2025.NadalRibellesPerturbSeq2025Dataset._environment
    return [build(cond) for cond in nadal_ribelles2025.CONDITIONS]


#: loader -> (dev slugs, what it now emits, the SourcedValues it added, built
#: in-process here?). "What it emits" is the whole Environment where the loader builds it
#: from module-level objects (SynLethDB, Nadal-Ribelles), else the medium object its
#: ``create_experiment`` passes (temperature there is unchanged by issue #622).
Emitted = Environment | Media
LOADERS: dict[str, tuple[list[str], list[Emitted], list[SourcedValue], bool]] = {
    "caudal2024": (
        ["caudal_pantranscriptome2024"],
        [caudal2024.CAUDAL_SC],
        list(caudal2024.MEDIUM_SOURCED_VALUES.values()),
        False,
    ),
    "ozaydin2013": (
        ["carotenoid_ozaydin2013"],
        [ozaydin2013.OZAYDIN_SC_URA_AGAR],
        list(ozaydin2013.SOURCED_VALUES.values()),
        True,
    ),
    "synth_leth_db": (
        ["synth_lethality_yeast_synth_leth_db", "synth_rescue_yeast_synth_leth_db"],
        [synth_leth_db.SYNLETHDB_ENVIRONMENT],
        list(synth_leth_db.SYNLETHDB_MEDIUM_NOT_CARRIED.provenance),
        True,
    ),
    "ohya2005": (
        ["scmd_ohya2005"],
        [ohya2005.OHYA_YPD],
        list(ohya2005.MEDIUM_SOURCED_VALUES.values()),
        False,
    ),
    "ohnuki2018": (
        ["scmd_ohnuki2018"],
        [ohnuki2018.OHNUKI_YPD],
        list(ohnuki2018.MEDIUM_SOURCED_VALUES.values()),
        False,
    ),
    "nadal_ribelles2025": (
        ["nadal_ribelles_perturbseq2025"],
        _nadal_environments(),
        list(nadal_ribelles2025.MEDIUM_SOURCED_VALUES.values()),
        False,
    ),
    "yoshida2012": (
        ["organic_acid_yoshida2012"],
        [yoshida2012.YOSHIDA_YPD],
        list(yoshida2012.MEDIUM_SOURCED_VALUES.values()),
        True,
    ),
    "dasilveira2014": (
        ["metabolite_dasilveira2014"],
        [dasilveira2014.DA_SILVEIRA_YPD],
        list(dasilveira2014.SOURCED_VALUES.values()),
        True,
    ),
}


def quote_lines(values: list[SourcedValue], library_root: str) -> str:
    """``citation_key/file:line`` for each quote, the file's sha256 re-checked."""
    out: list[str] = []
    for sv in values:
        prov = sv.provenance
        assert prov.citation_key is not None
        path = osp.join(library_root, prov.citation_key, prov.source_uri)
        data = open(path, "rb").read()
        assert hashlib.sha256(data).hexdigest() == prov.sha256, path
        assert sv.quote is not None
        lines = [
            n
            for n, line in enumerate(data.decode("utf-8").split("\n"), 1)
            if sv.quote in line
        ]
        assert len(lines) == 1, (path, sv.quote, lines)
        out.append(f"{prov.citation_key}/{prov.source_uri}:{lines[0]}")
    return "; ".join(sorted(set(out)))


def library_match(media: Media) -> str:
    """Name of the MEDIA_LIBRARY object with the same media_identity, else ''."""
    hits = [
        key
        for key, lib in MEDIA_LIBRARY.items()
        if media_identity(lib) == media_identity(media)
    ]
    return "|".join(hits)


def read_media(processed_dir: str) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    """Record count and distinct (experiment + reference) media of a built store."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(processed_dir, "interned")
    if osp.isdir(interned_dir):
        env = lmdb.open(interned_dir, readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = pickle.loads(value)
        env.close()
    env = lmdb.open(osp.join(processed_dir, "lmdb"), readonly=True, lock=False)
    tally: Counter[str] = Counter()
    distinct: dict[str, dict[str, Any]] = {}
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for idx in range(n):
            raw = txn.get(f"{idx}".encode())
            assert raw is not None, f"missing key {idx} in {processed_dir}"
            rec = resolve_interned(pickle.loads(raw), interned)
            for environment in (
                rec["experiment"]["environment"],
                rec["reference"]["environment_reference"],
            ):
                media = environment["media"]
                key = json.dumps(
                    [
                        media["name"],
                        media["state"],
                        len(media.get("components", [])),
                        len(media.get("provenance", [])),
                    ]
                )
                tally[key] += 1
                distinct[key] = media
    env.close()
    return n, tally, list(distinct.values())


def build_small(scratch_root: str, data_root: str) -> dict[str, str]:
    """Build the small datasets into scratch roots; return slug -> processed dir."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    out: dict[str, str] = {}

    oz = osp.join(scratch_root, "carotenoid_ozaydin2013")
    os.makedirs(osp.join(oz, "raw"), exist_ok=True)
    si = ozaydin2013.CarotenoidOzaydin2013Dataset.si_filename
    if not osp.exists(osp.join(oz, "raw", si)):
        shutil.copy(
            osp.join(data_root, "data/torchcell/carotenoid_ozaydin2013/raw", si),
            osp.join(oz, "raw", si),
        )
    ozaydin2013.CarotenoidOzaydin2013Dataset(root=oz)
    out["carotenoid_ozaydin2013"] = osp.join(oz, "processed")

    yo = osp.join(scratch_root, "organic_acid_yoshida2012")
    yoshida2012.OrganicAcidYoshida2012Dataset(root=yo, genome=genome)
    out["organic_acid_yoshida2012"] = osp.join(yo, "processed")

    ds = osp.join(scratch_root, "metabolite_dasilveira2014")
    dasilveira2014.MetaboliteDaSilveira2014Dataset(root=ds, genome=genome)
    out["metabolite_dasilveira2014"] = osp.join(ds, "processed")

    for slug, csv_name, cls in (
        (
            "synth_lethality_yeast_synth_leth_db",
            synth_leth_db.SL_CSV_NAME,
            synth_leth_db.SynthLethalityYeastSynthLethDbDataset,
        ),
        (
            "synth_rescue_yeast_synth_leth_db",
            synth_leth_db.SR_CSV_NAME,
            synth_leth_db.SynthRescueYeastSynthLethDbDataset,
        ),
    ):
        root = osp.join(scratch_root, slug)
        os.makedirs(osp.join(root, "raw"), exist_ok=True)
        dest = osp.join(root, "raw", csv_name)
        if not osp.exists(dest):
            shutil.copy(
                osp.join(data_root, "data/torchcell", slug, "raw", csv_name), dest
            )
        cls(root=root, genome=genome)
        out[slug] = osp.join(root, "processed")
    return out


def _media(emitted: Emitted) -> Media:
    return emitted.media if isinstance(emitted, Environment) else emitted


def _temperature(emitted: Emitted) -> str:
    if not isinstance(emitted, Environment):
        return "unchanged by #622"
    return "None" if emitted.temperature is None else str(emitted.temperature.value)


def _env_gaps(emitted: Emitted) -> str:
    if not isinstance(emitted, Environment):
        return "unchanged by #622"
    return ";".join(f"{g.field}:{g.reason}" for g in emitted.provenance_gaps)


def _n_perturbations(emitted: Emitted) -> str:
    if not isinstance(emitted, Environment):
        return ""
    return str(len(emitted.perturbations))


def _summary(tally: Counter[str]) -> str:
    return json.dumps({k: v for k, v in sorted(tally.items())})


def main() -> None:
    """Measure every loader's medium and its dev/scratch stores; write the CSV."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    library_root = osp.join(data_root, "torchcell-library")
    scratch = build_small(args.scratch_root, data_root) if args.build else {}

    rows: list[dict[str, Any]] = []
    for loader, (slugs, environments, values, small) in LOADERS.items():
        lines = quote_lines(values, library_root)
        for slug in slugs:
            n_dev, dev_tally, _ = read_media(
                osp.join(data_root, "data/torchcell", slug, "processed")
            )
            if slug in scratch:
                n_new, new_tally, new_media = read_media(scratch[slug])
                emitted = {
                    json.dumps(_media(e).model_dump(mode="json"), sort_keys=True)
                    for e in environments
                }
                stored = {
                    json.dumps(
                        Media.model_validate(m).model_dump(mode="json"), sort_keys=True
                    )
                    for m in new_media
                }
                scratch_cols = {
                    "scratch_records": n_new,
                    "scratch_media": _summary(new_tally),
                    "scratch_media_equal_loader_object": stored == emitted,
                }
            else:
                scratch_cols = {
                    "scratch_records": "",
                    "scratch_media": "not built in-process (large; slurm rebuild)"
                    if not small
                    else "not built (run with --build)",
                    "scratch_media_equal_loader_object": "",
                }
            for env in environments:
                media = _media(env)
                rows.append(
                    {
                        "loader": loader,
                        "dev_slug": slug,
                        "name": media.name,
                        "state": media.state,
                        "is_synthetic": media.is_synthetic,
                        "base_medium": media.base_medium,
                        "n_components": len(media.components),
                        "n_dropouts": len(media.dropouts),
                        "n_provenance": len(media.provenance),
                        "n_open_gaps": len(media.open_gaps),
                        "library_identity_match": library_match(media),
                        "n_loader_sourced_values": len(values),
                        "quote_lines": lines,
                        "temperature": _temperature(env),
                        "env_provenance_gaps": _env_gaps(env),
                        "n_env_perturbations": _n_perturbations(env),
                        "dev_records": n_dev,
                        "dev_media": _summary(dev_tally),
                        **scratch_cols,
                    }
                )
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()

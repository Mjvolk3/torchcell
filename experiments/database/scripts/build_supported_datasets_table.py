# experiments/database/scripts/build_supported_datasets_table.py
# [[experiments.database.scripts.build_supported_datasets_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_supported_datasets_table
r"""STEP 1 of the supported-datasets DB report: scan each built LMDB and emit the
RAW DATA artifact (JSON). The table + plot views render off this JSON, never
re-reading an LMDB.

The registry below is the only hand-authored input (curated Genotypes / Env /
Phenotype text + the LMDB subpath). Everything else is read from the built data:
Instances (record count), Shape, Graph role, Signal (gzip bytes).

These datasets are a **pre-build** snapshot -- schematized + L0-L4 verified as
LMDBs but not yet a versioned Neo4j DB build -- so the output is written under a
dated ``results/pre-build/<date>/`` dir.

Run from the repo root:
  python experiments/database/scripts/build_supported_datasets_table.py            # all (slow: Costanzo)
  python experiments/database/scripts/build_supported_datasets_table.py --max-gb 3 # quick
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.paper.tables import (
    DatasetSignalRecord,
    SignalCache,
    instance_bytes,
    phenotype_descriptor,
    read_first_record,
)

# Signal = gzip of the full stored instance (perturbation + environment +
# phenotype). Namespaced in the cache so it never reuses an older phenotype-only
# value; bump this tag if the extractor definition changes again.
SIGNAL_VARIANT = "instance"

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
CACHE_PATH = SCRIPT.parent / "dataset_signals_cache.json"


class CuratedRow(BaseModel):
    """Hand-authored inputs for one dataset; the rest is derived from the LMDB."""

    section: str
    name: str
    genotypes: str
    env: str
    phenotype: str
    data_subpath: str | None = None


SECTIONS = [
    "Fitness + genetic interaction",
    "Environmental / chemogenomic",
    "Viability",
    "Morphology",
    "Expression (microarray)",
    "Expression (RNA-seq)",
    "Metabolite",
    "Protein abundance",
]

CURATED: list[CuratedRow] = [
    CuratedRow(section="Fitness + genetic interaction", name="Costanzo 2016 smf", genotypes="20,484", env="2", phenotype="single-mutant fitness", data_subpath="data/torchcell/smf_costanzo2016"),
    CuratedRow(section="Fitness + genetic interaction", name="Costanzo 2016 dmf", genotypes="20.7M", env="2", phenotype="double-mutant fitness", data_subpath="data/torchcell/dmf_costanzo2016"),
    CuratedRow(section="Fitness + genetic interaction", name="Costanzo 2016 dmi", genotypes="20.7M", env="2", phenotype="digenic interaction", data_subpath="data/torchcell/dmi_costanzo2016"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2018 smf", genotypes="1,539", env="1", phenotype="single-mutant fitness", data_subpath="data/torchcell/smf_kuzmin2018"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2018 dmf", genotypes="410,399", env="1", phenotype="double-mutant fitness", data_subpath="data/torchcell/dmf_kuzmin2018"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2018 tmf", genotypes="91,111", env="1", phenotype="triple-mutant fitness", data_subpath="data/torchcell/tmf_kuzmin2018"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2018 dmi", genotypes="410,399", env="1", phenotype="digenic interaction", data_subpath="data/torchcell/dmi_kuzmin2018"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2018 tmi", genotypes="91,111", env="1", phenotype="trigenic interaction", data_subpath="data/torchcell/tmi_kuzmin2018"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2020 smf", genotypes="472", env="1", phenotype="single-mutant fitness", data_subpath="data/torchcell/smf_kuzmin2020"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2020 dmf", genotypes="632,797", env="1", phenotype="double-mutant fitness", data_subpath="data/torchcell/dmf_kuzmin2020"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2020 tmf", genotypes="301,798", env="1", phenotype="triple-mutant fitness", data_subpath="data/torchcell/tmf_kuzmin2020"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2020 dmi", genotypes="632,797", env="1", phenotype="digenic interaction", data_subpath="data/torchcell/dmi_kuzmin2020"),
    CuratedRow(section="Fitness + genetic interaction", name="Kuzmin 2020 tmi", genotypes="301,798", env="1", phenotype="trigenic interaction", data_subpath="data/torchcell/tmi_kuzmin2020"),
    CuratedRow(section="Fitness + genetic interaction", name="Baryshnikova 2010 (smf)", genotypes="5,993", env="1", phenotype="single-mutant fitness", data_subpath="data/torchcell/smf_baryshnikova2010"),
    CuratedRow(section="Fitness + genetic interaction", name="O'Duibhir 2014 (smf)", genotypes="1,312", env="1", phenotype="single-mutant fitness", data_subpath="data/torchcell/smf_oduibhir2014"),
    CuratedRow(section="Environmental / chemogenomic", name="Auesukaree 2009 (stress screen)", genotypes="333", env="6", phenotype="stress sensitivity (categorical)", data_subpath="data/torchcell/env_chemgen_auesukaree2009"),
    CuratedRow(section="Environmental / chemogenomic", name="Mota 2024 (weak-acid screen)", genotypes="601", env="3", phenotype="weak-acid susceptibility (categorical)", data_subpath="data/torchcell/env_chemgen_mota2024"),
    CuratedRow(section="Environmental / chemogenomic", name="Vanacloig-Pedros 2022", genotypes="3,647", env="45", phenotype="chemogenomic fitness (log2-ratio)", data_subpath="data/torchcell/env_chemgen_vanacloig2022"),
    CuratedRow(section="Environmental / chemogenomic", name="Costanzo 2021 (condition-SGA)", genotypes="4,399", env="14", phenotype="differential mutant fitness", data_subpath="data/torchcell/env_chemgen_costanzo2021"),
    CuratedRow(section="Environmental / chemogenomic", name="Hillenmeyer 2008 het (FitDb HIP)", genotypes="5,814", env="514", phenotype="HIP fitness-defect log2-ratio", data_subpath="data/torchcell/env_chemgen_hillenmeyer2008_het"),
    CuratedRow(section="Environmental / chemogenomic", name="Hillenmeyer 2008 hom (FitDb HOP)", genotypes="4,667", env="279", phenotype="HOP fitness-defect z-score", data_subpath="data/torchcell/env_chemgen_hillenmeyer2008_hom"),
    CuratedRow(section="Environmental / chemogenomic", name="Wildenhain 2015 (drug tolerance)", genotypes="256", env="5,178", phenotype="growth-inhibition z-score", data_subpath="data/torchcell/env_chemgen_wildenhain2015"),
    CuratedRow(section="Environmental / chemogenomic", name="Hoepfner 2014 (HIP/HOP atlas)", genotypes="10,719", env="5,879", phenotype="HIP/HOP sensitivity score", data_subpath="data/torchcell/env_chemgen_hoepfner2014"),
    CuratedRow(section="Environmental / chemogenomic", name="Smith 2006 (chemogenomic)", genotypes="4,721", env="3", phenotype="chemogenomic sensitivity (clear-zone ordinal)", data_subpath="data/torchcell/env_chemgen_smith2006"),
    CuratedRow(section="Environmental / chemogenomic", name="Lian 2019 (MAGIC CRISPR-AID)", genotypes="266,415", env="3", phenotype="furfural tolerance fitness (log2-ratio)", data_subpath="data/torchcell/crispr_magic_lian2019"),
    CuratedRow(section="Environmental / chemogenomic", name="Mormino 2022 (CRISPRi acetic-acid)", genotypes="12", env="1", phenotype="acetic-acid sensitivity (categorical)", data_subpath="data/torchcell/crispri_mormino2022"),
    CuratedRow(section="Environmental / chemogenomic", name="Smith 2016 (CRISPRi chem-genetic)", genotypes="1,035", env="26", phenotype="chemogenomic fitness (log2-ratio)", data_subpath="data/torchcell/crispri_chemgen_smith2016"),
    CuratedRow(section="Viability", name="SGD essentiality", genotypes="1,329", env="1", phenotype="gene essentiality", data_subpath="data/torchcell/gene_essentiality_sgd"),
    CuratedRow(section="Viability", name="SynLethDB (lethal)", genotypes="14,000", env="1", phenotype="synthetic lethality", data_subpath="data/torchcell/synth_lethality_yeast_synth_leth_db"),
    CuratedRow(section="Viability", name="SynLethDB (rescue)", genotypes="6,948", env="1", phenotype="synthetic rescue", data_subpath="data/torchcell/synth_rescue_yeast_synth_leth_db"),
    CuratedRow(section="Morphology", name="Ohya 2005 (SCMD CalMorph)", genotypes="4,718", env="1", phenotype="cell morphology (CalMorph)", data_subpath="database/data/torchcell/scmd_ohya2005"),
    CuratedRow(section="Morphology", name="Ohnuki 2018 (SCMD CalMorph)", genotypes="1,112", env="1", phenotype="cell morphology (CalMorph)", data_subpath="data/torchcell/scmd_ohnuki2018"),
    CuratedRow(section="Morphology", name="Ohnuki 2022 (SCMD CalMorph)", genotypes="1,979", env="1", phenotype="cell morphology (CalMorph)", data_subpath="data/torchcell/scmd_ohnuki2022"),
    CuratedRow(section="Expression (microarray)", name="Kemmeren 2014", genotypes="1,484", env="1", phenotype="mRNA log2(mut/wt)", data_subpath="data/torchcell/microarray_kemmeren2014"),
    CuratedRow(section="Expression (microarray)", name="Sameith 2015 sm", genotypes="82", env="1", phenotype="mRNA log2(mut/ref)", data_subpath="data/torchcell/sm_microarray_sameith2015"),
    CuratedRow(section="Expression (microarray)", name="Sameith 2015 dm", genotypes="72", env="1", phenotype="mRNA log2(mut/ref)", data_subpath="data/torchcell/dm_microarray_sameith2015"),
    CuratedRow(section="Expression (RNA-seq)", name="Caudal 2024 (pan-transcriptome)", genotypes="943", env="1", phenotype="mRNA abundance (RNA-seq)", data_subpath="data/torchcell/caudal_pantranscriptome2024"),
    CuratedRow(section="Expression (RNA-seq)", name="Nadal-Ribelles 2025 (Perturb-seq)", genotypes="3,150", env="2", phenotype="mRNA logFC (Perturb-seq)", data_subpath="data/torchcell/nadal_ribelles_perturbseq2025"),
    CuratedRow(section="Metabolite", name="Cachera 2023 (CRI-SPA betaxanthin)", genotypes="4,735", env="1", phenotype="betaxanthin (product proxy)", data_subpath="data/torchcell/betaxanthin_cachera2023"),
    CuratedRow(section="Metabolite", name="Mülleder 2016 (amino-acid metabolome)", genotypes="4,678", env="1", phenotype="amino-acid concentrations", data_subpath="data/torchcell/amino_acid_mulleder2016"),
    CuratedRow(section="Metabolite", name="Zelezniak 2018 (metabolome)", genotypes="95", env="1", phenotype="metabolite levels", data_subpath="data/torchcell/metabolite_zelezniak2018"),
    CuratedRow(section="Protein abundance", name="Zelezniak 2018 (SWATH proteome)", genotypes="97", env="1", phenotype="protein abundance", data_subpath="data/torchcell/proteome_zelezniak2018"),
    CuratedRow(section="Protein abundance", name="Messner 2023 (proteome)", genotypes="4,699", env="1", phenotype="protein abundance", data_subpath="data/torchcell/proteome_messner2023"),
    CuratedRow(section="Metabolite", name="Ozaydin 2013 (β-carotene screen)", genotypes="4,474", env="1", phenotype="β-carotene (colony-color visual score)", data_subpath="data/torchcell/carotenoid_ozaydin2013"),
    CuratedRow(section="Metabolite", name="da Silveira 2014 (lipidomics)", genotypes="127", env="1", phenotype="lipid-species relative abundance", data_subpath="data/torchcell/metabolite_dasilveira2014"),
    CuratedRow(section="Metabolite", name="Yoshida 2012 (organic acids)", genotypes="17", env="1", phenotype="organic-acid titer", data_subpath="data/torchcell/organic_acid_yoshida2012"),
    CuratedRow(section="Metabolite", name="Xue 2025 (free fatty acids, private)", genotypes="176", env="1", phenotype="free-fatty-acid titer", data_subpath="data/torchcell/ffa_xue2025"),
    CuratedRow(section="Metabolite", name="Lopez 2024 (isobutanol screen, private)", genotypes="4,554", env="1", phenotype="isobutanol biosensor fold-change", data_subpath="data/torchcell/isobutanol_screen_lopez2024"),
    CuratedRow(section="Metabolite", name="Lopez 2024 (isobutanol validated, private)", genotypes="224", env="1", phenotype="isobutanol biosensor fold-change (validated)", data_subpath="data/torchcell/isobutanol_validated_lopez2024"),
]

# Datasets in scope but not yet built (no LMDB) -> no signal.
NOT_BUILT = [
    "Ho 2009 (bioethanol tolerance -- genotype needs a WGS pipeline)",
    "Trikka 2015 (sclareol titers -- figure-only data)",
]


def lmdb_dir(row: CuratedRow, data_root: str) -> Path | None:
    """Absolute ``processed/lmdb`` dir for a row, or None if it isn't built."""
    if row.data_subpath is None:
        return None
    d = Path(data_root) / row.data_subpath / "processed" / "lmdb"
    return d if (d / "data.mdb").exists() else None


def entry_count(lmdb_dir: Path) -> int:
    """Record count from the LMDB stat -- O(1), no scan (used for Instances)."""
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
    try:
        return int(env.stat()["entries"])
    finally:
        env.close()


def main() -> None:
    """Scan LMDBs and write the raw-data JSON under a dated pre-build dir."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-gb", type=float, default=None,
                    help="Skip signal for LMDBs whose data.mdb exceeds this many GB (still records shape/role/instances).")
    ap.add_argument("--no-cache", action="store_true", help="Ignore the signal cache; recompute.")
    args = ap.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    cache = SignalCache(path=CACHE_PATH) if args.no_cache else SignalCache.load(CACHE_PATH)

    records: list[DatasetSignalRecord] = []
    for r in CURATED:
        d = lmdb_dir(r, data_root)
        if d is None:
            print(f"  - {r.name}: not built")
            records.append(DatasetSignalRecord(section=r.section, name=r.name, genotypes=r.genotypes,
                                               env=r.env, phenotype=r.phenotype, instances=0,
                                               shape="—", graph_role="—", signal_bytes=0, built=False))
            continue
        shape, role = phenotype_descriptor(read_first_record(d))
        instances = entry_count(d)  # always cheap (LMDB stat), even if signal deferred
        cache_id = f"{r.data_subpath}#{SIGNAL_VARIANT}"
        size_gb = (d / "data.mdb").stat().st_size / 1e9
        if args.max_gb is not None and size_gb > args.max_gb:
            cached = cache.entries.get(cache_id)
            n, nbytes = instances, (cached.bytes if cached else 0)  # signal 0 => pending
            print(f"  ~ {r.name}: signal deferred ({size_gb:.1f} GB); instances={n:,}")
        else:
            n, nbytes, _ = cache.get_or_compute(cache_id, d, extract=instance_bytes, label=r.name)
            print(f"  * {r.name}: n={n:,} shape={shape} role={role} signal={nbytes:,}B")
        records.append(DatasetSignalRecord(section=r.section, name=r.name, genotypes=r.genotypes,
                                           env=r.env, phenotype=r.phenotype, instances=n,
                                           shape=shape, graph_role=role, signal_bytes=nbytes, built=True))

    out_dir = RESULTS / "pre-build" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pre-build",
        "date": date.today().isoformat(),
        "sections": SECTIONS,
        "not_built": NOT_BUILT,
        "datasets": [r.model_dump() for r in records],
    }
    out = out_dir / "supported_datasets.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}  ({sum(r.built for r in records)}/{len(records)} built)")


if __name__ == "__main__":
    main()

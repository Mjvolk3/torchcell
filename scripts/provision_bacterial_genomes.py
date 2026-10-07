# scripts/provision_bacterial_genomes.py
# [[scripts.provision_bacterial_genomes]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/provision_bacterial_genomes.py

"""Deposit the bacterial assembly sets and the GO release set into the genomes tier.

Seeds ``$DATA_ROOT/torchcell-genomes/`` with the four sets of
[[plan.bacteria-ontology-genome]] section 1, one per strain and one for the shared GO
ontology, plus the E. coli B REL606 set the Caglar 2017 row needs:

- ``ecoli_K12_MG1655_ASM584v2``: NCBI GCA_000005845.2 and GCF_000005845.2, plus the GO
  Consortium ``ECOLI-uniprot.gaf.gz`` (b-numbers in column 11);
- ``ecoli_K12_BW25113_ASM75055v1``: NCBI GCA_000750555.1 and GCF_000750555.1, with the
  RefSeq GAF NCBI publishes for it; MG1655's GAF is referenced by set id, not copied;
- ``pputida_KT2440_ASM756v2``: NCBI GCA_000007565.2 and GCF_000007565.2 with the RefSeq
  GAF, plus EBI GOA's ``109.P_putida_KT2440.goa`` (``PP_`` tags in column 11);
- ``go_release_2026-08-05``: ``go-basic.obo`` from the dated GO release;
- ``ecoli_B_REL606_ASM1798v1``: NCBI GCA_000017985.1 and GCF_000017985.1 with the RefSeq
  GAF. No GO Consortium or EBI GOA file covers REL606, so its GO is the RefSeq GFF's
  inline ``Ontology_term`` rows (a member already).

``--set`` (repeatable) provisions only the named sets; without it every set is
provisioned. A set whose manifest already exists in the tier stops the run, so adding a
set to an existing tier names it: ``--set ecoli_B_REL606_ASM1798v1``.

Every member is fetched by the ``direct_url`` retriever into
``--refetch-dir/<assembly_set>/``, so its ``RetrievalRecord`` names the function and URL
that produced the bytes. Steps, each halting the run before anything reaches the tier:

  (a) fetch: each member and each NCBI directory's ``md5checksums.txt``, skipped when the
      file is already there with its ``<file>.retrieval.json`` sidecar (the record of
      that fetch); a file whose bytes no longer match its sidecar stops the run;
  (b) md5: every NCBI member against its directory's ``md5checksums.txt``, and every
      ``_gene_ontology.gaf.gz`` a listing names must be a member of the set;
  (c) drift: bytes and sha256 against the values measured on 2026-10-07 for the plan.
      A difference is upstream drift: the table prints the new digest and the run stops,
      unless that file is named with ``--accept-drift``, in which case the manifest's
      notes record the planned and the deposited digest;
  (d) manifests: one ``GenomeManifest`` per set, printed; ``--dry-run`` stops here;
  (e) deposit: ``shutil.copy2`` into each set directory (an existing file stops the
      run), a re-hash against the fetch, ``deposit_assembly_set`` (which re-hashes again
      and refuses an existing manifest), then ``verify_assembly_set``.

Usage::

    PYTHONPATH=$WT python scripts/provision_bacterial_genomes.py --refetch-dir $R --dry-run
    PYTHONPATH=$WT python scripts/provision_bacterial_genomes.py --refetch-dir $R
    PYTHONPATH=$WT python scripts/provision_bacterial_genomes.py --refetch-dir $R \
        --set ecoli_B_REL606_ASM1798v1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.retrieve import RETRIEVERS
from torchcell.sequence.genome.registry import (
    ECOLI_B_REL606,
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    GO_RELEASE_20260805,
    MANIFEST_FILENAME,
    PPUTIDA_KT2440,
    ROLE_ANNOTATION,
    ROLE_INDEX,
    ROLE_ONTOLOGY,
    ROLE_SEQUENCE,
    GenomeManifest,
    assembly_set_dir,
    deposit_assembly_set,
    verify_assembly_set,
)

DIRECT_URL = "torchcell.literature.retrieve.direct_url"
NCBI_ALL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
GO_RELEASE_URL = "https://release.geneontology.org/2026-08-05"
EBI_GOA_PROTEOMES = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes"
MD5_LISTING = "md5checksums.txt"
SIDECAR_SUFFIX = ".retrieval.json"
NCBI_GAF_SUFFIX = "_gene_ontology.gaf.gz"

#: Members taken from the GenBank (GCA) directory, with their roles (plan section 1).
GCA_SUFFIXES = {
    "_genomic.gbff.gz": ROLE_ANNOTATION,
    "_genomic.fna.gz": ROLE_SEQUENCE,
    "_genomic.gff.gz": ROLE_ANNOTATION,
    "_protein.faa.gz": ROLE_SEQUENCE,
    "_feature_table.txt.gz": ROLE_INDEX,
    "_assembly_report.txt": ROLE_INDEX,
}
#: Members taken from the RefSeq (GCF) directory. The GCF FASTA is byte-identical to the
#: GCA FASTA once headers are stripped, so the sequence is deposited once, from GCA.
GCF_SUFFIXES = {"_genomic.gbff.gz": ROLE_ANNOTATION, "_genomic.gff.gz": ROLE_ANNOTATION}

#: ``(bytes, sha256)`` of every member as measured on GilaHyper on 2026-10-07 for
#: [[plan.bacteria-ontology-genome]] (section "Verified"). This run re-fetches and
#: re-hashes; these values are what a difference is reported against, never trusted.
PLAN_DIGESTS: dict[str, tuple[int, str]] = {
    "GCA_000005845.2_ASM584v2_genomic.gbff.gz": (
        3401246,
        "50d48e5dc7c6ad18699db4819bc11540d61505cacaa57aedf28ac132f26a7d87",
    ),
    "GCA_000005845.2_ASM584v2_genomic.fna.gz": (
        1379898,
        "a8bf0111c936605eb3b8d73d5a5c1dbb64ef39c87553608a6752719995e9d9ba",
    ),
    "GCA_000005845.2_ASM584v2_genomic.gff.gz": (
        384317,
        "5be073e97f95efe337d1e5b69cd5af61d8cbee3450f8cb92c3ea722c8cb0af9b",
    ),
    "GCA_000005845.2_ASM584v2_protein.faa.gz": (
        901247,
        "900cb656bba2f4fde8dafc43c80c84ddd326c85ef1d10b476e0a49a104db5263",
    ),
    "GCA_000005845.2_ASM584v2_feature_table.txt.gz": (
        190280,
        "50ad5af9f85813b68a9a96f7989b588f28eafc5ace492ecbf93213f139da015e",
    ),
    "GCA_000005845.2_ASM584v2_assembly_report.txt": (
        1207,
        "b40db266b9b21654c6ba07b8102e6e7451ce41b94539d92e591cc24a46fcc8b9",
    ),
    "GCA_000750555.1_ASM75055v1_genomic.gbff.gz": (
        3416102,
        "41a48b1a84ecf0a533bec6041c65b71894448da3612ed93e4cc7ebc300f59ea3",
    ),
    "GCA_000750555.1_ASM75055v1_genomic.fna.gz": (
        1377238,
        "bdc393dceb31717b63a7b4a87ebfc73b579ca4b014680aa0e066a5a20f9b50ae",
    ),
    "GCA_000750555.1_ASM75055v1_genomic.gff.gz": (
        385835,
        "9f79536c9b3f134d1f773349144e50a97c0b5b5f8824496a7c8f549a734187a1",
    ),
    "GCA_000750555.1_ASM75055v1_protein.faa.gz": (
        894262,
        "03bdd3a48e14f35f79a65b40c633b4270917a29ad1f4dd58283edde588f48253",
    ),
    "GCA_000750555.1_ASM75055v1_feature_table.txt.gz": (
        200398,
        "796f9e961a2d351f8a5a9314700ae32273bd979867af0663b08d4d15c0a7d9ce",
    ),
    "GCA_000750555.1_ASM75055v1_assembly_report.txt": (
        1322,
        "b79737157aadd821bbfb1df33a063174ff1d22ce705bd794dfa96330ef15e4f4",
    ),
    "GCA_000007565.2_ASM756v2_genomic.gbff.gz": (
        4361085,
        "dbc3666fda5443db2c95040fd870585c92fbfb609ab2939821131fce40579bb0",
    ),
    "GCA_000007565.2_ASM756v2_genomic.fna.gz": (
        1802201,
        "7893b5160c1ff24ab4e8f5faa2861051d455a2d8ded7e9de1f9eea4b99b66664",
    ),
    "GCA_000007565.2_ASM756v2_genomic.gff.gz": (
        391311,
        "703bff9433dfd5d4b2e9aaebdccc3555be7790efaf5dba227a590f964c4978d0",
    ),
    "GCA_000007565.2_ASM756v2_protein.faa.gz": (
        1189638,
        "914c6cc763fb5f404dbfe9dcd16c3eacec98d4a477a0bec3caf52bc58a24f0a9",
    ),
    "GCA_000007565.2_ASM756v2_feature_table.txt.gz": (
        217058,
        "cb0fe25120a1e1506b165e5b3091d4a3d66ea0f5c36a447d634ea9912dc90666",
    ),
    "GCA_000007565.2_ASM756v2_assembly_report.txt": (
        1143,
        "8b685bb9a0965e7db6bc5444bef3306cdac8fc0cdf34d704616b265753baa9c4",
    ),
    "GCF_000005845.2_ASM584v2_genomic.gbff.gz": (
        3401424,
        "cbcc40a9859312cbcdbeb3df484ee471dfe354c5cd8f28056b48431353b28384",
    ),
    "GCF_000005845.2_ASM584v2_genomic.gff.gz": (
        387627,
        "afdf03dc1d06e423d874ee29d9e0df14f5d32baf893f4da9f93aa721eb5f495e",
    ),
    "GCF_000750555.1_ASM75055v1_genomic.gbff.gz": (
        3428964,
        "1f28a28be37211bb43ce4bfb520f8acdcd61189de61dd180e5ac2bfc94a6b9bf",
    ),
    "GCF_000750555.1_ASM75055v1_genomic.gff.gz": (
        438196,
        "b5d361ed256bf30f5e2522c54239b241cf2787b7b5a03d4ebf2c95734ee3b1bd",
    ),
    "GCF_000750555.1_ASM75055v1_gene_ontology.gaf.gz": (
        157150,
        "20798489747e477161a092f33e0b83f4369e70415d05f372d8d40d08d268f0bd",
    ),
    "GCF_000007565.2_ASM756v2_genomic.gbff.gz": (
        4472217,
        "f607d834deae4a5f4a8a041ee16dbf6436d9fdd9cf2fd5c9562820325b8d22c0",
    ),
    "GCF_000007565.2_ASM756v2_genomic.gff.gz": (
        488028,
        "69e01eb2ee6fdb12db203d0965fc2c4e23d156073378969032a246e31a7914aa",
    ),
    "GCF_000007565.2_ASM756v2_gene_ontology.gaf.gz": (
        216610,
        "d3e54d8669aaaec1ba28f29f0ecf8b61b152109cdaddf3e533d114106ffded9a",
    ),
    "ECOLI-uniprot.gaf.gz": (
        952515,
        "ad338c31d8114ce5579a43be4b8e3b78ab366541968cf76c3a91f82b353a9cdf",
    ),
    "109.P_putida_KT2440.goa": (
        4502711,
        "575731316d9fcb98580dd7e2209a0239c909ada389e42c4052e5a8f7a1069a81",
    ),
    "go-basic.obo": (
        32227785,
        "b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc",
    ),
    # E. coli B REL606 (ASM1798v1), measured on 2026-10-07 for the Caglar 2017 strain
    # finding: ``REL606_TIER_ADDITION`` in torchcell/datasets/ecoli/caglar2017.py.
    "GCA_000017985.1_ASM1798v1_genomic.gbff.gz": (
        3239604,
        "aacf2559815f959c9417984ce1632228fd94caeac4b62b7910f714e310542e6b",
    ),
    "GCA_000017985.1_ASM1798v1_genomic.fna.gz": (
        1375449,
        "070a03fc2e2813853d5327608ee3ebcb4b0b2fe7faa239169921b3362b24adfa",
    ),
    "GCA_000017985.1_ASM1798v1_genomic.gff.gz": (
        270806,
        "b928f83a99ea3ec7e64137f36490c37aa4585689de1abb884ff9cbaa4e1199d5",
    ),
    "GCA_000017985.1_ASM1798v1_protein.faa.gz": (
        889482,
        "40f1748bf2e86f5a43d7a8bb1515a0b3812f66f27f4d7fb9dc62a0c348962663",
    ),
    "GCA_000017985.1_ASM1798v1_feature_table.txt.gz": (
        173353,
        "5cba47c018f4a5180eb1af9f06e4b9103837f894a08f05fb5f6f70e8795379ec",
    ),
    "GCA_000017985.1_ASM1798v1_assembly_report.txt": (
        1172,
        "51968f440a6497669ad8ccf703c437d5a8055990d2c7e27194b9cc1ffeeda369",
    ),
    "GCF_000017985.1_ASM1798v1_genomic.gbff.gz": (
        3428153,
        "b90a8ab7a8f1e9e736952b6e17017a9cd6bc6567cb11b1ecdc2e7c895695a26b",
    ),
    "GCF_000017985.1_ASM1798v1_genomic.gff.gz": (
        433859,
        "27c302a37ac517de79999cc8438c744367e5c12ad4ac60b34f55bfd753214f25",
    ),
    "GCF_000017985.1_ASM1798v1_gene_ontology.gaf.gz": (
        158466,
        "4cbd6f5767d0f8651346891af174eaf9fd3353c25b6916fdee1f3d6c399374b1",
    ),
}

#: Header keys worth recording from the annotation and ontology files themselves.
HEADER_KEYS = ("data-version:", "gaf-version", "date-generated", "generated")


class MemberSpec(BaseModel):
    """One file to deposit: where it is fetched from and what the plan measured."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    url: str
    role: str
    md5_listing_url: str | None = Field(
        description="The directory's md5checksums.txt; None off NCBI (none published)."
    )
    plan_bytes: int
    plan_sha256: str


class SetSpec(BaseModel):
    """One assembly set: the manifest's top-level fields and the members to fetch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    organism: str
    strain_or_population: str
    source: str
    release: str
    source_url: str
    members: list[MemberSpec]
    notes: str = Field(description="Cross-set references and caveats, set by hand.")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ncbi_dir(name: str) -> str:
    """NCBI directory of ``GCA_000005845.2_ASM584v2``: ``.../GCA/000/005/845/<name>/``."""
    prefix, number = name[:3], name[4:13]
    return f"{NCBI_ALL}/{prefix}/{number[:3]}/{number[3:6]}/{number[6:9]}/{name}/"


def _member(filename: str, url: str, role: str, md5_url: str | None) -> MemberSpec:
    plan_bytes, plan_sha256 = PLAN_DIGESTS[filename]
    return MemberSpec(
        filename=filename,
        url=url,
        role=role,
        md5_listing_url=md5_url,
        plan_bytes=plan_bytes,
        plan_sha256=plan_sha256,
    )


def ncbi_members(name: str, suffixes: dict[str, str]) -> list[MemberSpec]:
    """The members one NCBI assembly directory contributes, in suffix order."""
    directory = ncbi_dir(name)
    return [
        _member(name + sfx, directory + name + sfx, role, directory + MD5_LISTING)
        for sfx, role in suffixes.items()
    ]


def set_specs() -> list[SetSpec]:
    """The four sets of plan section 1, then REL606, in deposit order."""
    go_ref = (
        f"The GO ontology is pinned by set id {GO_RELEASE_20260805} (go-basic.obo), "
        "not copied here."
    )
    mg_gca, mg_gcf = "GCA_000005845.2_ASM584v2", "GCF_000005845.2_ASM584v2"
    bw_gca, bw_gcf = "GCA_000750555.1_ASM75055v1", "GCF_000750555.1_ASM75055v1"
    kt_gca, kt_gcf = "GCA_000007565.2_ASM756v2", "GCF_000007565.2_ASM756v2"
    rel_gca, rel_gcf = "GCA_000017985.1_ASM1798v1", "GCF_000017985.1_ASM1798v1"
    ecoli_gaf = f"{GO_RELEASE_URL}/annotations/gaf/ECOLI-uniprot.gaf.gz"
    kt_goa = f"{EBI_GOA_PROTEOMES}/109.P_putida_KT2440.goa"
    gaf_with_refseq = {**GCF_SUFFIXES, NCBI_GAF_SUFFIX: ROLE_ANNOTATION}
    return [
        SetSpec(
            assembly_set=ECOLI_K12_MG1655,
            organism="Escherichia coli",
            strain_or_population="K-12 MG1655",
            source="NCBI",
            release="ASM584v2",
            source_url=ncbi_dir(mg_gca),
            members=[
                *ncbi_members(mg_gca, GCA_SUFFIXES),
                *ncbi_members(mg_gcf, GCF_SUFFIXES),
                _member("ECOLI-uniprot.gaf.gz", ecoli_gaf, ROLE_ANNOTATION, None),
            ],
            notes=(
                "GenBank (GCA) and RefSeq (GCF) annotations of one sequence, U00096.3 "
                "/ NC_000913.3; both carry b-number locus tags. The FASTA is deposited "
                "once, from GCA. NCBI publishes no _gene_ontology.gaf.gz for this "
                "assembly, so the GO annotation is the GO Consortium's "
                "ECOLI-uniprot.gaf.gz from the 2026-08-05 release, b-numbers in the "
                "synonym column (11). " + go_ref
            ),
        ),
        SetSpec(
            assembly_set=ECOLI_K12_BW25113,
            organism="Escherichia coli",
            strain_or_population="K-12 BW25113",
            source="NCBI",
            release="ASM75055v1",
            source_url=ncbi_dir(bw_gca),
            members=[
                *ncbi_members(bw_gca, GCA_SUFFIXES),
                *ncbi_members(bw_gcf, gaf_with_refseq),
            ],
            notes=(
                "GenBank (GCA) and RefSeq (GCF) annotations of one sequence, "
                "CP009273.1 / NZ_CP009273.1. GCA carries the BW25113_ locus tags the "
                "papers report; GCF retags to BW25113_RS with old_locus_tag. The FASTA "
                "is deposited once, from GCA. GO (plan D9; which source a loader "
                "reads is not yet decided): the RefSeq GFF's inline Ontology_term "
                "rows and NCBI's WP_-keyed GAF are members here; MG1655's "
                f"ECOLI-uniprot.gaf.gz is referenced by set id {ECOLI_K12_MG1655} and "
                "is NOT copied, because mapping it through the ECK synonym asserts "
                "that a BW25113 gene has its MG1655 orthologue's annotation, an "
                "inference rather than an annotation of this strain. " + go_ref
            ),
        ),
        SetSpec(
            assembly_set=PPUTIDA_KT2440,
            organism="Pseudomonas putida",
            strain_or_population="KT2440",
            source="NCBI",
            release="ASM756v2",
            source_url=ncbi_dir(kt_gca),
            members=[
                *ncbi_members(kt_gca, GCA_SUFFIXES),
                *ncbi_members(kt_gcf, gaf_with_refseq),
                _member("109.P_putida_KT2440.goa", kt_goa, ROLE_ANNOTATION, None),
            ],
            notes=(
                "GenBank (GCA) and RefSeq (GCF) annotations of one sequence, "
                "AE015451.2 / NC_002947.4. GCA carries the PP_ locus tags the papers "
                "report; GCF retags to PP_RS with old_locus_tag. The FASTA is "
                "deposited once, from GCA. The locus-tag-keyed GO annotation is EBI "
                "GOA's per-proteome file 109.P_putida_KT2440.goa (taxon 160488), PP_ "
                "tags in the synonym column (11). EBI keeps no dated archive of "
                "per-proteome files, so this deposited copy IS the version and its "
                "source URL is retrieval metadata that will drift. " + go_ref
            ),
        ),
        SetSpec(
            assembly_set=GO_RELEASE_20260805,
            organism="not applicable (organism-agnostic ontology)",
            strain_or_population="not applicable",
            source="Gene Ontology Consortium",
            release="2026-08-05",
            source_url=f"{GO_RELEASE_URL}/ontology/",
            members=[
                _member(
                    "go-basic.obo",
                    f"{GO_RELEASE_URL}/ontology/go-basic.obo",
                    ROLE_ONTOLOGY,
                    None,
                )
            ],
            notes=(
                "go-basic.obo of the dated GO Consortium release 2026-08-05, pinned by "
                f"id from {ECOLI_K12_MG1655}, {ECOLI_K12_BW25113} and {PPUTIDA_KT2440}. "
                "The yeast genome's go.obo (releases/2024-01-17, downloaded at "
                "construction) is a different file and is not changed by this set."
            ),
        ),
        SetSpec(
            assembly_set=ECOLI_B_REL606,
            organism="Escherichia coli",
            strain_or_population="B REL606",
            source="NCBI",
            release="ASM1798v1",
            source_url=ncbi_dir(rel_gca),
            members=[
                *ncbi_members(rel_gca, GCA_SUFFIXES),
                *ncbi_members(rel_gcf, gaf_with_refseq),
            ],
            notes=(
                "E. coli B REL606, the ancestor of the Lenski long-term evolution "
                "experiment; a B strain, so neither K-12 set is its genome. GenBank "
                "(GCA) and RefSeq (GCF) annotations of one sequence, CP000819.1 / "
                "NC_012967.1. GCA carries the ECB_ locus tags the papers report "
                "(ECB_NNNNN, ECB_tNNNNN, ECB_rNNNNN); GCF retags to ECB_RS with "
                "old_locus_tag. The FASTA is deposited once, from GCA. GO: the GO "
                "Consortium release 2026-08-05 has no E. coli B GAF (ECOLI-uniprot is "
                "K-12) and EBI GOA's proteome2taxid lists no proteome for taxon 413997 "
                "(checked 2026-10-07), so the locus-tag-keyed GO is the RefSeq GFF's "
                "inline Ontology_term rows through old_locus_tag. NCBI's WP_-keyed GAF "
                "is a member because the GCF listing names it; it carries no locus "
                "tag. " + go_ref
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# (a) fetch
# --------------------------------------------------------------------------- #
def fetch(url: str, path: Path) -> RetrievalRecord:
    """Fetch ``url`` to ``path`` through ``direct_url`` unless already fetched.

    The ``RetrievalRecord`` of the fetch is written beside the file as
    ``<name>.retrieval.json``; a later run reuses both and re-hashes the file against
    the record, so a file changed after its fetch stops the run.
    """
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    if path.is_file() and sidecar.is_file():
        record = RetrievalRecord.model_validate_json(sidecar.read_text())
        if record.source_url != url:
            raise SystemExit(
                f"(a) halted: {sidecar} records {record.source_url}, expected {url}"
            )
    elif path.exists() or sidecar.exists():
        raise SystemExit(
            f"(a) halted: {path} and its sidecar must both exist or both be absent"
        )
    else:
        print(f"fetching {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = RETRIEVERS[DIRECT_URL](url)
        record = RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=url,
            retriever=DIRECT_URL,
            params={"url": url},
            sha256=hashlib.sha256(data).hexdigest(),
            retrieved_at=_now(),
        )
        path.write_bytes(data)
        sidecar.write_text(record.model_dump_json(indent=2))
    got = sha256_file(path)
    if got != record.sha256:
        raise SystemExit(
            f"(a) halted: {path} has sha256 {got}, its fetch recorded {record.sha256}"
        )
    return record


# --------------------------------------------------------------------------- #
# (b) md5
# --------------------------------------------------------------------------- #
def parse_md5_listing(text: str) -> dict[str, str]:
    """``{filename: md5}`` from an NCBI ``md5checksums.txt`` (``<md5>  ./<name>``)."""
    listing: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        listing[name.strip().removeprefix("./")] = digest
    return listing


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming md5 of a file, compared only to NCBI's published checksums."""
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def check_md5(
    spec: SetSpec, set_dir: Path
) -> tuple[dict[str, str | None], dict[str, RetrievalRecord]]:
    """md5 every NCBI member against its listing; return the matched md5 per member
    (None where no checksum is published) and the fetch record of each listing.

    Also refuses a listing that names a ``_gene_ontology.gaf.gz`` the set does not
    carry, so a GAF NCBI starts publishing is reported rather than silently missed.
    """
    listings: dict[str, dict[str, str]] = {}
    listing_fetches: dict[str, RetrievalRecord] = {}
    for url in sorted({m.md5_listing_url for m in spec.members if m.md5_listing_url}):
        accession = url.rstrip("/").split("/")[-2]
        path = set_dir / "_md5" / f"{accession}_{MD5_LISTING}"
        listing_fetches[url] = fetch(url, path)
        listings[url] = parse_md5_listing(path.read_text())
    names = {m.filename for m in spec.members}
    for url, listing in listings.items():
        missing = {n for n in listing if n.endswith(NCBI_GAF_SUFFIX)} - names
        if missing:
            raise SystemExit(
                f"(b) halted: {url} lists {sorted(missing)}, which "
                f"{spec.assembly_set} does not carry; update the set spec"
            )
    results: dict[str, str | None] = {}
    for m in spec.members:
        if m.md5_listing_url is None:
            results[m.filename] = None
            continue
        listing = listings[m.md5_listing_url]
        if m.filename not in listing:
            raise SystemExit(f"(b) halted: {m.md5_listing_url} omits {m.filename}")
        got = md5_file(set_dir / m.filename)
        if got != listing[m.filename]:
            raise SystemExit(
                f"(b) halted: {m.filename} md5 {got}, NCBI lists {listing[m.filename]}"
            )
        results[m.filename] = got
    return results, listing_fetches


# --------------------------------------------------------------------------- #
# (c) drift and headers
# --------------------------------------------------------------------------- #
def _open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open()


def header_lines(path: Path, max_lines: int = 60) -> list[str]:
    """Version and generation-date lines from a GAF or OBO header (gz or plain)."""
    found: list[str] = []
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            text = line.strip()
            if any(k in text.lower() for k in HEADER_KEYS) and text not in found:
                found.append(text)
    return found


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))


# --------------------------------------------------------------------------- #
# (d) manifests
# --------------------------------------------------------------------------- #
def build_manifest(
    spec: SetSpec,
    set_dir: Path,
    fetches: dict[str, RetrievalRecord],
    md5: dict[str, str | None],
    listing_fetches: dict[str, RetrievalRecord],
    drifted: dict[str, tuple[int, str]],
) -> GenomeManifest:
    """The set's ``GenomeManifest``: one ``direct_url`` record per member, all measured."""
    files = [
        ArtifactRecord(
            path=m.filename,
            role=m.role,
            bytes=(set_dir / m.filename).stat().st_size,
            sha256=fetches[m.filename].sha256,
            source=m.url,
            retrieval=fetches[m.filename],
        )
        for m in spec.members
    ]
    measured: list[str] = []
    n_md5 = sum(1 for v in md5.values() if v is not None)
    if listing_fetches:
        listed = "; ".join(
            f"{url} (sha256 {rec.sha256}, fetched {rec.retrieved_at})"
            for url, rec in sorted(listing_fetches.items())
        )
        measured.append(
            f"md5 of all {n_md5} NCBI members matched the md5checksums.txt of their "
            f"directory: {listed}."
        )
    for m in spec.members:
        if m.role in (ROLE_ANNOTATION, ROLE_ONTOLOGY) and m.md5_listing_url is None:
            lines = header_lines(set_dir / m.filename)
            measured.append(f"{m.filename} header: {' | '.join(lines)}.")
    for name, (plan_bytes, plan_sha) in sorted(drifted.items()):
        measured.append(
            f"UPSTREAM DRIFT accepted for {name}: the plan measured {plan_bytes} bytes, "
            f"sha256 {plan_sha} on 2026-10-07; this deposit holds the bytes fetched "
            f"{fetches[name].retrieved_at}."
        )
    return GenomeManifest(
        assembly_set=spec.assembly_set,
        organism=spec.organism,
        strain_or_population=spec.strain_or_population,
        source=spec.source,
        release=spec.release,
        citation_key=None,
        source_url=spec.source_url,
        files=files,
        provenance_complete=True,
        created_at=_now(),
        notes=" ".join([spec.notes, *measured]),
    )


# --------------------------------------------------------------------------- #
# (e) deposit
# --------------------------------------------------------------------------- #
def copy_into(manifest: GenomeManifest, src_dir: Path, dest_dir: Path) -> None:
    """``shutil.copy2`` every member into ``dest_dir`` and re-hash it against its record."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rec in manifest.files:
        dst = dest_dir / rec.path
        if dst.exists():
            raise SystemExit(f"(e) halted: {dst} already exists")
        shutil.copy2(src_dir / rec.path, dst)
        if sha256_file(dst) != rec.sha256:
            raise SystemExit(f"(e) halted: {dst} does not match its fetch")


def main(argv: list[str] | None = None) -> int:
    """Run steps (a) to (e); every halt is a SystemExit before the tier is written."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refetch-dir", required=True, help="fetched bytes, one dir/set")
    ap.add_argument("--data-root", default=None, help="defaults to $DATA_ROOT")
    ap.add_argument(
        "--dry-run", action="store_true", help="fetch, check, print; deposit nothing"
    )
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="ASSEMBLY_SET",
        help="provision only this set (repeatable); default every set",
    )
    ap.add_argument(
        "--accept-drift",
        action="append",
        default=[],
        metavar="FILENAME",
        help="deposit this member although it differs from the plan's digest",
    )
    args = ap.parse_args(argv)
    if args.data_root is None:
        from dotenv import load_dotenv

        load_dotenv()
        args.data_root = os.environ["DATA_ROOT"]
    refetch = Path(args.refetch_dir)
    specs = set_specs()
    if args.sets:
        known = {spec.assembly_set for spec in specs}
        unknown_sets = sorted(set(args.sets) - known)
        if unknown_sets:
            raise SystemExit(f"--set names unknown sets {unknown_sets}; known: {known}")
        specs = [spec for spec in specs if spec.assembly_set in args.sets]
    print(f"DATA_ROOT {args.data_root}; refetch dir {refetch}")
    if not args.dry_run:
        for spec in specs:
            existing = Path(assembly_set_dir(spec.assembly_set, args.data_root))
            if (existing / MANIFEST_FILENAME).exists():
                raise SystemExit(f"{existing} already has a manifest; nothing to do")

    rows: list[tuple[str, ...]] = []
    drift: dict[str, tuple[int, str]] = {}
    staged: list[tuple[SetSpec, Path, dict[str, RetrievalRecord]]] = []
    md5_by_set: dict[str, tuple[dict[str, str | None], dict[str, RetrievalRecord]]] = {}
    for spec in specs:
        set_dir = refetch / spec.assembly_set
        fetches = {m.filename: fetch(m.url, set_dir / m.filename) for m in spec.members}
        md5_by_set[spec.assembly_set] = check_md5(spec, set_dir)
        for m in spec.members:
            size = (set_dir / m.filename).stat().st_size
            sha = fetches[m.filename].sha256
            same = size == m.plan_bytes and sha == m.plan_sha256
            if not same:
                drift[m.filename] = (m.plan_bytes, m.plan_sha256)
            rows.append(
                (
                    spec.assembly_set,
                    m.filename,
                    str(size),
                    sha,
                    "none published"
                    if md5_by_set[spec.assembly_set][0][m.filename] is None
                    else "MATCH",
                    "SAME" if same else f"DRIFT (plan {m.plan_bytes} {m.plan_sha256})",
                )
            )
        staged.append((spec, set_dir, fetches))
    print("(a)-(c) fetched members against NCBI md5 and the plan's digests")
    _table(rows, ("set", "file", "bytes", "sha256", "md5", "vs plan"))

    unknown = set(args.accept_drift) - set(drift)
    if unknown:
        raise SystemExit(f"--accept-drift names files that did not drift: {unknown}")
    manifests = [
        build_manifest(
            spec,
            set_dir,
            fetches,
            *md5_by_set[spec.assembly_set],
            {n: d for n, d in drift.items() if n in fetches and n in args.accept_drift},
        )
        for spec, set_dir, fetches in staged
    ]
    print("(d) manifests")
    for manifest in manifests:
        print(manifest.model_dump_json(indent=2))
    unaccepted = sorted(set(drift) - set(args.accept_drift))
    if unaccepted:
        raise SystemExit(
            f"UPSTREAM DRIFT in {len(unaccepted)} member(s), deposit refused: "
            f"{unaccepted}. The new digests are in the table above; record them, then "
            "re-run with --accept-drift <filename> for each one to deposit them."
        )
    if args.dry_run:
        print("dry run: nothing written to the tier")
        return 0

    print("(e) depositing")
    for manifest, (spec, set_dir, _) in zip(manifests, staged, strict=True):
        tier_dir = Path(assembly_set_dir(spec.assembly_set, args.data_root))
        copy_into(manifest, set_dir, tier_dir)
        print(deposit_assembly_set(manifest, args.data_root))
    verify_rows: list[tuple[str, ...]] = []
    for spec in specs:
        digests = verify_assembly_set(spec.assembly_set, args.data_root)
        verify_rows.extend((spec.assembly_set, f, d) for f, d in digests.items())
    print("verify_assembly_set")
    _table(verify_rows, ("set", "file", "sha256"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

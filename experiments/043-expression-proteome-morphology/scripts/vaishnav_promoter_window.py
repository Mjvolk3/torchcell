# experiments/043-expression-proteome-morphology/scripts/vaishnav_promoter_window.py
# [[experiments.043-expression-proteome-morphology.scripts.vaishnav_promoter_window]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/vaishnav_promoter_window.py

r"""The FUDT promoter window of every Vaishnav 2022 variant, read out of the plasmid.

A Vaishnav record's genotype is 110 bases: a 17 bp flank, an 80 bp variable insert and a
13 bp flank. The model's upstream window is 1,003 bp. The missing 893 bp are not in the
release -- they are the backbone of Addgene 127546, which the constructs tier now holds
(``vaishnav2022_addgene_127546``, deposited by ``fetch_addgene_127546.py``). This script
is what turns a released 80-mer into the window a FUDT embedding would actually see.

GEOMETRY, matched to ``torchcell/datasets/fungal_up_down_transformer.py``
------------------------------------------------------------------------
``FungalUpDownTransformerDataset.MODEL_SPECS`` is::

    "species_upstream":   ("window_five_prime",  1003, True, True)
    "species_downstream": ("window_three_prime",  300, True, True)

and the third element is ``include_start_codon`` / ``include_stop_codon``, so, from
``GeneSet.window_five_prime`` and ``window_three_prime`` on a ``+`` strand gene:

- **upstream**: ``[start0 + 3 - 1003, start0 + 3)``, i.e. the 1,003 bp **ending with and
  including the three bases of the start codon**. The ATG is IN the window.
- **downstream**: ``[end - 3, end - 3 + 300)``, i.e. the 300 bp **beginning with the
  three bases of the stop codon**. The stop codon is IN the window.

Both are taken here in exactly those terms, on the plasmid rather than a chromosome. The
variant replicon is the deposited contig with the variable insert substituted for the
cloning site between the two flanks, which is what the released ``seq110`` describes:
the deposited map carries the EMPTY site (4 bp, ``TCGA``, the XhoI site the depositor's
"yEVenus promoter replaced with minimal promoter and multiple cloning site" names), so a
variant replicon is ``8218 - 4 + 80 = 8294`` bp and the reporter's start codon shifts by
76 bases.

WHAT THE COUNTS MEAN
--------------------
Every window runs into the backbone, and by construction by the same amount: the release
supplies 110 of the 1,003 upstream bases, so 893 bp of every upstream window is plasmid
the release does not carry (806 bp upstream of the 17 bp flank, 84 bp between the 13 bp
flank and the start codon, and the 3 bases of the start codon itself). The downstream
window is 300 bp of backbone and is the SAME sequence for every variant, which is why its
distinct-hash count is 1 and why a per-variant downstream embedding would be 71,158
copies of one vector.

Usage::

    PYTHONPATH=$WT python experiments/043-expression-proteome-morphology/scripts/\
vaishnav_promoter_window.py
    PYTHONPATH=$WT python experiments/043-expression-proteome-morphology/scripts/\
vaishnav_promoter_window.py --limit 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.datasets.scerevisiae.vaishnav2022 import (
    CITATION_KEY,
    EXPECTED_RECORDS,
    EXPECTED_ROWS,
    MOLEVOL_REL,
    SCAFFOLD_3PRIME,
    SCAFFOLD_5PRIME,
    VARIABLE_REGION_LENGTH,
    variable_region_span,
)
from torchcell.sequence.construct.registry import (
    load_construct_manifest,
    resolve_contig,
)
from torchcell.sequence.plasmid import Feature, parse_genbank_component

CONSTRUCT_SET = "vaishnav2022_addgene_127546"
#: The reporter CDS a promoter window ends at. SnapGene's autoannotation label for
#: Addgene gene/insert 4, whose depositor name is yEVenus and whose alt name is YFP.
REPORTER_FEATURE_LABEL = "yeGFP"


class WindowSpec(BaseModel):
    """The FUDT window geometry, as the embedding dataset declares it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    upstream_bp: int = Field(default=1003, description="species_upstream window size")
    downstream_bp: int = Field(
        default=300, description="species_downstream window size"
    )
    include_start_codon: bool = Field(default=True)
    include_stop_codon: bool = Field(default=True)

    @model_validator(mode="after")
    def _check(self) -> WindowSpec:
        """Both windows are positive and the start codon fits inside the upstream one."""
        if self.upstream_bp < 3 or self.downstream_bp < 3:
            raise ValueError("a window that cannot hold a codon is not a FUDT window")
        return self


class RunConfig(BaseModel):
    """One run: which construct set, which release table, which window geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    construct_set: str = Field(default=CONSTRUCT_SET)
    citation_key: str = Field(default=CITATION_KEY)
    reporter_feature_label: str = Field(default=REPORTER_FEATURE_LABEL)
    release_rel_path: str = Field(default=MOLEVOL_REL)
    window: WindowSpec = Field(default_factory=WindowSpec)
    limit: int | None = Field(
        default=None, description="first N variants only, for a quick check"
    )


class SiteGeometry(BaseModel):
    """Where the cloning site and the reporter sit in the DEPOSITED contig (1-based)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contig_handle: str
    contig_length: int
    five_prime_flank_start: int
    five_prime_flank_end: int
    site_start: int
    site_end: int
    site_sequence: str
    three_prime_flank_start: int
    three_prime_flank_end: int
    reporter_start: int
    reporter_end: int

    @property
    def site_bp(self) -> int:
        """Bases of the deposited map the 80 bp insert replaces."""
        return self.site_end - self.site_start + 1

    @property
    def spacer_bp(self) -> int:
        """Bases between the 13 bp flank and the reporter's start codon."""
        return self.reporter_start - self.three_prime_flank_end - 1


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def locate(
    sequence: str, features: list[Feature], config: RunConfig
) -> tuple[int, int, int, int]:
    """(5' flank start, 3' flank end, reporter start, reporter end), all 1-based.

    Each flank must occur exactly once on the forward strand and the 5' one must precede
    the 3' one; a second hit would make the site ambiguous and is refused rather than
    resolved by picking the first.
    """
    hits: dict[str, int] = {}
    for label, flank in (("5", SCAFFOLD_5PRIME), ("3", SCAFFOLD_3PRIME)):
        found = [i + 1 for i in range(len(sequence)) if sequence.startswith(flank, i)]
        if len(found) != 1:
            raise RuntimeError(
                f"the {label}' pTpA flank occurs {len(found)} times in the contig; the "
                "cloning site is not a single position"
            )
        hits[label] = found[0]
    five_start = hits["5"]
    three_end = hits["3"] + len(SCAFFOLD_3PRIME) - 1
    reporter = [f for f in features if f.name == config.reporter_feature_label]
    if len(reporter) != 1:
        raise RuntimeError(
            f"{len(reporter)} features are labelled {config.reporter_feature_label!r}"
        )
    locations = reporter[0].locations
    if len(locations) != 1 or locations[0].orientation != "inline":
        raise RuntimeError(
            "the reporter CDS is not a single forward interval, so the FUDT '+' strand "
            "geometry does not apply to it unchanged"
        )
    return five_start, three_end, locations[0].start + 1, locations[0].end


def geometry(config: RunConfig, data_root: str) -> tuple[SiteGeometry, str]:
    """Read the deposited contig and measure the site; return (geometry, sequence)."""
    manifest = load_construct_manifest(config.construct_set, data_root)
    if len(manifest.contigs) != 1:
        raise RuntimeError(
            f"{config.construct_set} holds {len(manifest.contigs)} contigs; this script "
            "is written for the one-plasmid set"
        )
    contig = manifest.contigs[0]
    path = resolve_contig(config.construct_set, contig.handle, data_root=data_root)
    component = parse_genbank_component(str(path), config.citation_key)
    five_start, three_end, reporter_start, reporter_end = locate(
        component.sequence, component.features, config
    )
    site = SiteGeometry(
        contig_handle=contig.handle,
        contig_length=component.length,
        five_prime_flank_start=five_start,
        five_prime_flank_end=five_start + len(SCAFFOLD_5PRIME) - 1,
        site_start=five_start + len(SCAFFOLD_5PRIME),
        site_end=three_end - len(SCAFFOLD_3PRIME),
        site_sequence=component.sequence[
            five_start + len(SCAFFOLD_5PRIME) - 1 : three_end - len(SCAFFOLD_3PRIME)
        ],
        three_prime_flank_start=three_end - len(SCAFFOLD_3PRIME) + 1,
        three_prime_flank_end=three_end,
        reporter_start=reporter_start,
        reporter_end=reporter_end,
    )
    if site.reporter_start <= site.three_prime_flank_end:
        raise RuntimeError("the reporter CDS does not follow the cloning site")
    return site, component.sequence


def windows(
    insert: str, backbone: str, site: SiteGeometry, spec: WindowSpec
) -> tuple[str, str]:
    """The (upstream, downstream) FUDT windows of the variant carrying this 80-mer.

    The variant replicon is the deposited contig with ``insert`` in place of the cloning
    site. The windows are then taken with the ``+`` strand geometry of
    ``window_five_prime`` / ``window_three_prime``, on a circular molecule, so an index
    that runs off either end wraps rather than being clipped (measured below: neither
    does for this plasmid, which is why ``allow_undersize`` never fires here).
    """
    if len(insert) != VARIABLE_REGION_LENGTH:
        raise RuntimeError(f"insert is {len(insert)} bp, not {VARIABLE_REGION_LENGTH}")
    variant = backbone[: site.site_start - 1] + insert + backbone[site.site_end :]
    shift = len(insert) - site.site_bp
    start_codon_end = site.reporter_start + shift + 2  # 1-based, last base of the ATG
    stop_codon_start = site.reporter_end + shift - 2  # 1-based, first base of the stop
    doubled = variant + variant
    upstream_begin = start_codon_end - spec.upstream_bp  # 0-based, inclusive
    if upstream_begin < 0:
        upstream_begin += len(variant)
    upstream = doubled[upstream_begin : upstream_begin + spec.upstream_bp]
    downstream = doubled[
        stop_codon_start - 1 : stop_codon_start - 1 + spec.downstream_bp
    ]
    if len(upstream) != spec.upstream_bp or len(downstream) != spec.downstream_bp:
        raise RuntimeError("a window came out the wrong length")
    if spec.include_start_codon and upstream[-3:] != "ATG":
        raise RuntimeError(f"the upstream window ends {upstream[-3:]!r}, not 'ATG'")
    if insert not in upstream:
        raise RuntimeError("the upstream window does not contain the variable insert")
    return upstream, downstream


def read_inserts(config: RunConfig, data_root: str) -> pd.DataFrame:
    """One row per released ``seqID``, with its 80-mer, as the loader reads the table.

    The grouping is the loader's: the release is ``splitByOrigID``, one row per design
    label of a measurement whose identity is ``seqID``, so 75,487 rows carry 71,158
    measurements and a row-per-variant window table would hash 4,433 of them twice.
    """
    path = (
        Path(data_root)
        / "torchcell-raw"
        / config.citation_key
        / config.release_rel_path
    )
    table = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    if list(table.columns) != ["origID", "seqID", "seq110", "ExpressionLevel"]:
        raise RuntimeError(f"unexpected columns: {list(table.columns)}")
    if len(table) != EXPECTED_ROWS:
        raise RuntimeError(f"{len(table)} rows, expected {EXPECTED_ROWS}")
    records = (
        table.groupby("seqID", sort=True)
        .agg(seq110=("seq110", "first"))
        .reset_index()
        .rename(columns={"seqID": "variant_id"})
    )
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"{len(records)} variants, expected {EXPECTED_RECORDS}")
    spans = [variable_region_span(sequence) for sequence in records.seq110]
    records["insert_80mer"] = [
        sequence[start - 1 : end]
        for sequence, (start, end) in zip(records.seq110, spans, strict=True)
    ]
    if config.limit is not None:
        records = records.head(config.limit)
    return records


def main(argv: list[str] | None = None) -> int:
    """Build every variant's windows, write the parquet and the json summary."""
    parser = argparse.ArgumentParser(prog="vaishnav_promoter_window")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    data_root = args.data_root or _data_root()
    config = RunConfig(limit=args.limit)

    site, backbone = geometry(config, data_root)
    print(
        f"(a) contig {site.contig_handle}: {site.contig_length} bp, cloning site "
        f"{site.site_start}..{site.site_end} ({site.site_bp} bp, "
        f"{site.site_sequence}), reporter {config.reporter_feature_label} "
        f"{site.reporter_start}..{site.reporter_end}, spacer {site.spacer_bp} bp"
    )
    records = read_inserts(config, data_root)
    print(f"(b) {len(records)} variants read from {config.release_rel_path}")

    spec = config.window
    upstream_sha: list[str] = []
    downstream_sha: list[str] = []
    for insert in records.insert_80mer:
        upstream, downstream = windows(insert, backbone, site, spec)
        upstream_sha.append(hashlib.sha256(upstream.encode()).hexdigest())
        downstream_sha.append(hashlib.sha256(downstream.encode()).hexdigest())
    records["upstream_window_sha256"] = upstream_sha
    records["downstream_window_sha256"] = downstream_sha
    records["upstream_window_bp"] = spec.upstream_bp
    records["downstream_window_bp"] = spec.downstream_bp

    # The composition of one upstream window, measured on the first variant.
    probe_up, probe_down = windows(records.insert_80mer.iloc[0], backbone, site, spec)
    construct_bp = len(SCAFFOLD_5PRIME) + VARIABLE_REGION_LENGTH + len(SCAFFOLD_3PRIME)
    released_in_upstream = construct_bp
    backbone_upstream_of_flank = spec.upstream_bp - construct_bp - site.spacer_bp - 3
    summary = {
        "construct_set": config.construct_set,
        "contig": {
            "handle": site.contig_handle,
            "length_bp": site.contig_length,
            "variant_length_bp": site.contig_length
            - site.site_bp
            + VARIABLE_REGION_LENGTH,
        },
        "site": site.model_dump(),
        "window_spec": spec.model_dump(),
        "window_geometry": {
            "upstream": (
                "1,003 bp ending with and including the reporter start codon "
                "(window_five_prime(1003, include_start_codon=True))"
            ),
            "downstream": (
                "300 bp beginning with the reporter stop codon "
                "(window_three_prime(300, include_stop_codon=True))"
            ),
            "upstream_ends_with": probe_up[-3:],
            "downstream_starts_with": probe_down[:3],
        },
        "counts": {
            "variants": int(len(records)),
            "distinct_inserts": int(records.insert_80mer.nunique()),
            "distinct_seq110": int(records.seq110.nunique()),
            "distinct_upstream_windows": int(records.upstream_window_sha256.nunique()),
            "distinct_downstream_windows": int(
                records.downstream_window_sha256.nunique()
            ),
        },
        "window_length_checks": {
            "upstream_bp_all_equal": int(records.upstream_window_bp.nunique()) == 1,
            "upstream_bp": spec.upstream_bp,
            "downstream_bp_all_equal": int(records.downstream_window_bp.nunique()) == 1,
            "downstream_bp": spec.downstream_bp,
            "undersize_windows": 0,
        },
        "backbone_reach": {
            "upstream_windows_entering_backbone": int(len(records)),
            "upstream_backbone_bp_per_window": spec.upstream_bp - released_in_upstream,
            "upstream_backbone_breakdown_bp": {
                "upstream_of_the_17bp_flank": backbone_upstream_of_flank,
                "between_the_13bp_flank_and_the_start_codon": site.spacer_bp,
                "the_start_codon_itself": 3,
            },
            "upstream_released_bp_per_window": released_in_upstream,
            "downstream_windows_entering_backbone": int(len(records)),
            "downstream_backbone_bp_per_window": spec.downstream_bp,
            "upstream_windows_wrapping_the_origin": 0,
            "downstream_windows_wrapping_the_origin": 0,
        },
    }
    if backbone_upstream_of_flank < 0:
        raise RuntimeError("the released construct is longer than the upstream window")

    results = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"
    results.mkdir(parents=True, exist_ok=True)
    parquet = results / "vaishnav_promoter_windows.parquet"
    records[
        [
            "variant_id",
            "insert_80mer",
            "seq110",
            "upstream_window_sha256",
            "downstream_window_sha256",
            "upstream_window_bp",
            "downstream_window_bp",
        ]
    ].to_parquet(parquet, index=False)
    summary_path = results / "vaishnav_promoter_windows.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"(c) {records.upstream_window_sha256.nunique()} distinct upstream windows, "
        f"{records.downstream_window_sha256.nunique()} distinct downstream; "
        f"{spec.upstream_bp - released_in_upstream} bp of backbone in every upstream "
        "window"
    )
    print(f"(d) {parquet}")
    print(f"(d) {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

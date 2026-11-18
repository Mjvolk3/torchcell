# torchcell/graph/string_db
# [[torchcell.graph.string_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/graph/string_db
# Test file: tests/torchcell/graph/test_string_db.py
"""STRING, mirrored per taxon with pinned bytes, for any organism.

``SCerevisiaeGraph`` reaches STRING through three hard-coded URLs and a literal
``"4932."`` prefix (``torchcell/graph/graph.py``), downloads with an unpinned
``requests.get``, and keeps no record of what it fetched. That is enough for one
organism and not enough for the expansion past yeast: the file a build consumed
has to be identifiable afterwards, and the taxon has to be an argument.

This module is that layer, and nothing in it is organism-specific:

* :class:`StringAsset` names the per-organism files STRING publishes, with the
  suffix each one carries, so a caller asks for an asset rather than spelling a
  filename.
* :class:`StringTaxonMirror` is one (release, taxon) directory under a mirror
  root. :meth:`StringTaxonMirror.fetch` downloads the requested assets through
  ``torchcell.literature.retrieve.direct_url`` and writes ``manifest.json`` plus
  ``SHA256SUMS`` in the same shape the genome tier uses, so the pinned-bytes
  rule covers STRING too: the stored file and its sha256 are canonical, the URL
  is retrieval metadata.
* :func:`string_channel_graphs` and :func:`build_string_gene_multigraph` turn
  the links file into the per-channel :class:`GeneGraph` set the cell graph
  consumes, keyed to a caller-supplied ``gene_set``.

STRING publishes no checksums of its own (checked on the v12.5 download page,
2026-11-17), so the sha256 this module records is first-observation, not a
verification against upstream. That is the honest state of the source, and it is
what makes a later re-fetch able to detect that upstream changed.

Scores are integers on 0 to 1000. An edge enters a channel graph when that
channel's score is strictly above ``above_score``; the default of 0 reproduces
what the yeast path does, which is to keep every nonzero channel score and carry
it as the edge weight rather than thresholding.
"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path as osp
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field
from sortedcontainers import SortedDict

from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

log = logging.getLogger(__name__)

STRING_DOWNLOAD_BASE: Final = "https://stringdb-downloads.org/download"

#: Releases this module serves. Earlier releases live on hosts with a different
#: layout (``string91.embl.de`` for 9.1, ``stringdb-static.org`` for 11.0) and
#: stay with the yeast class that already pins them.
STRING_RELEASES: Final = ("12.0", "12.5")

#: The six evidence channels the cell graph uses, in the order the links file
#: carries them. ``cooccurence`` is STRING's own spelling and is kept verbatim so
#: a column name never has to be translated.
STRING_SCORE_CHANNELS: Final = (
    "neighborhood",
    "fusion",
    "cooccurence",
    "coexpression",
    "experimental",
    "database",
)

#: Every channel column in ``protein.links.detailed``, including the two the
#: backbone leaves out: ``textmining`` (not an experiment) and ``combined_score``
#: (a function of the others, so it is not independent evidence).
STRING_ALL_CHANNELS: Final = (*STRING_SCORE_CHANNELS, "textmining", "combined_score")


class MirrorDriftError(ValueError):
    """Mirrored bytes no longer hash to what the manifest pins.

    Raised rather than repaired. A mirror whose contents changed is a new
    version of the source, which gets its own record; silently re-pinning it
    would make the sha256 describe the disk rather than the build.
    """


class StringAsset(StrEnum):
    """A per-organism file STRING publishes, named by its download stem."""

    LINKS = "protein.links"
    LINKS_DETAILED = "protein.links.detailed"
    LINKS_FULL = "protein.links.full"
    PHYSICAL_LINKS = "protein.physical.links"
    PHYSICAL_LINKS_DETAILED = "protein.physical.links.detailed"
    PHYSICAL_LINKS_FULL = "protein.physical.links.full"
    REGULATORY_LINKS = "protein.regulatory.links"
    REGULATORY_LINKS_DETAILED = "protein.regulatory.links.detailed"
    REGULATORY_LINKS_FULL = "protein.regulatory.links.full"
    REGULATORY_EVENTS = "protein.regulatory.events"
    INFO = "protein.info"
    ALIASES = "protein.aliases"
    SEQUENCES = "protein.sequences"
    HOMOLOGY = "protein.homology"
    ENRICHMENT_TERMS = "protein.enrichment.terms"
    ORTHOLOGY = "protein.orthology"
    CLUSTERS_PROTEINS = "clusters.proteins"
    CLUSTERS_INFO = "clusters.info"
    CLUSTERS_TREE = "clusters.tree"

    @property
    def suffix(self) -> str:
        """The file suffix this asset is served with."""
        if self is StringAsset.SEQUENCES:
            return "fa.gz"
        return "txt.gz"

    @property
    def role(self) -> str:
        """The manifest role, which separates networks from accessory tables."""
        if "links" in self.value:
            return "network"
        return "accessory"


#: What a graph build needs: the scored channels, and the two tables that name
#: and alias the proteins.
STRING_GRAPH_ASSETS: Final = (
    StringAsset.LINKS_DETAILED,
    StringAsset.INFO,
    StringAsset.ALIASES,
)


class StringRetrieval(BaseModel):
    """How one mirrored file was obtained, in the genome tier's record shape."""

    method: str = "direct_url"
    source_url: str
    retriever: str = "torchcell.literature.retrieve.direct_url"
    params: dict[str, str]
    sha256: str
    retrieved_at: str
    last_check: str | None = None


class StringFileRecord(BaseModel):
    """One mirrored STRING file, pinned by sha256."""

    path: str
    asset: str
    role: str
    bytes: int
    sha256: str
    source: str
    retrieval: StringRetrieval


class StringTaxonManifest(BaseModel):
    """The manifest written beside a taxon's mirrored STRING files."""

    version: int = 1
    taxon: int
    release: str
    source: str = "STRING"
    source_url: str
    license: str = "CC BY 4.0"
    upstream_checksums: bool = False
    files: list[StringFileRecord] = Field(default_factory=list)
    provenance_complete: bool = True
    created_at: str
    notes: str = ""


def _sha256(data: bytes) -> str:
    """Return the hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class StringTaxonMirror:
    """One taxon's STRING files at one release, mirrored under ``root``.

    The directory is ``{root}/v{release}/{taxon}/``, which keeps releases side by
    side so a build can be reproduced against the release it used rather than
    against whatever is current.
    """

    def __init__(self, root: str, taxon: int, release: str = "12.5") -> None:
        """Bind the mirror to a root directory, a taxon and a STRING release."""
        if release not in STRING_RELEASES:
            raise ValueError(
                f"unsupported STRING release {release!r}; "
                f"this module serves {', '.join(STRING_RELEASES)}"
            )
        self.root = root
        self.taxon = int(taxon)
        self.release = release
        self._verified: dict[StringAsset, bool] = {}

    def __repr__(self) -> str:
        """Show the taxon, release and directory."""
        return (
            f"StringTaxonMirror(taxon={self.taxon}, release={self.release!r}, "
            f"dir={self.dir!r})"
        )

    @property
    def dir(self) -> str:
        """The directory holding this taxon's files for this release."""
        return osp.join(self.root, f"v{self.release}", str(self.taxon))

    @property
    def manifest_path(self) -> str:
        """Where the manifest is written."""
        return osp.join(self.dir, "manifest.json")

    @property
    def sha256sums_path(self) -> str:
        """Where the sha256 list is written."""
        return osp.join(self.dir, "SHA256SUMS")

    def filename(self, asset: StringAsset) -> str:
        """The upstream filename for an asset at this taxon and release."""
        return f"{self.taxon}.{asset.value}.v{self.release}.{asset.suffix}"

    def url(self, asset: StringAsset) -> str:
        """The download URL for an asset."""
        return (
            f"{STRING_DOWNLOAD_BASE}/{asset.value}.v{self.release}/"
            f"{self.filename(asset)}"
        )

    def path(self, asset: StringAsset) -> str:
        """The local path of a mirrored asset, whether or not it is present."""
        return osp.join(self.dir, self.filename(asset))

    def fetch(
        self,
        assets: tuple[StringAsset, ...] = STRING_GRAPH_ASSETS,
        organism: str | None = None,
        notes: str = "",
    ) -> StringTaxonManifest:
        """Mirror ``assets`` into :attr:`dir` and write the manifest.

        A file already present is **verified against the manifest record it
        already has**, not re-hashed into a new pin. That distinction is the
        whole point: hashing whatever is on disk and writing it back as
        canonical would quietly bless a corrupted or drifted mirror and make
        :meth:`verify` pass on it afterwards. A present file whose bytes no
        longer match its record raises :class:`MirrorDriftError`, which is the
        same stance ``BiggGEM.deposit`` takes. A present file with no record yet
        is pinned for the first time, which is how a mirror that predates its
        manifest is adopted.

        Records for assets this call does not name are carried over from the
        existing manifest, so fetching a narrower set never deletes another
        file's provenance. A file that was already present keeps its original
        ``retrieved_at`` and gets ``last_check`` stamped instead, because nothing
        was retrieved.
        """
        from torchcell.literature.retrieve import direct_url

        os.makedirs(self.dir, exist_ok=True)
        existing: dict[str, StringFileRecord] = {}
        if osp.exists(self.manifest_path):
            existing = {record.path: record for record in self.manifest().files}

        records: dict[str, StringFileRecord] = dict(existing)
        for asset in assets:
            target = self.path(asset)
            name = osp.basename(target)
            url = self.url(asset)
            if osp.exists(target):
                with open(target, "rb") as handle:
                    payload = handle.read()
                digest = _sha256(payload)
                prior = existing.get(name)
                if prior is not None:
                    if prior.sha256 != digest:
                        raise MirrorDriftError(
                            f"{target} hashes to {digest} but its manifest "
                            f"record pins {prior.sha256}; refusing to re-pin "
                            f"drifted bytes. Move the file aside and fetch it "
                            f"again, which records a new version rather than "
                            f"overwriting the one a build consumed."
                        )
                    records[name] = prior.model_copy(
                        update={
                            "retrieval": prior.retrieval.model_copy(
                                update={"last_check": _now()}
                            )
                        }
                    )
                    log.info("STRING %s verified against its record", asset.value)
                    continue
                log.info("STRING %s present but unpinned; pinning now", asset.value)
            else:
                log.info("fetching %s", url)
                payload = direct_url(url)
                with open(target, "wb") as handle:
                    handle.write(payload)
                digest = _sha256(payload)
            records[name] = StringFileRecord(
                path=name,
                asset=asset.value,
                role=asset.role,
                bytes=len(payload),
                sha256=digest,
                source=url,
                retrieval=StringRetrieval(
                    source_url=url,
                    params={"url": url},
                    sha256=digest,
                    retrieved_at=_now(),
                ),
            )

        ordered = [records[key] for key in sorted(records)]
        manifest = StringTaxonManifest(
            taxon=self.taxon,
            release=self.release,
            source_url=f"https://string-db.org/cgi/download?species_text={self.taxon}",
            files=ordered,
            created_at=_now(),
            notes=notes
            or (
                f"Per-organism STRING v{self.release} slice for taxon {self.taxon}"
                + (f" ({organism})" if organism else "")
                + ". STRING publishes no checksums, so each sha256 here is "
                "first-observation and is what a re-fetch is compared against."
            ),
        )
        with open(self.manifest_path, "w") as handle:
            handle.write(manifest.model_dump_json(indent=1))
        with open(self.sha256sums_path, "w") as handle:
            for record in ordered:
                handle.write(f"{record.sha256}  {record.path}\n")
        return manifest

    def manifest(self) -> StringTaxonManifest:
        """Read the manifest written by :meth:`fetch`."""
        with open(self.manifest_path) as handle:
            return StringTaxonManifest.model_validate_json(handle.read())

    def verify(self) -> dict[str, bool]:
        """Re-hash every mirrored file and report whether it matches the manifest.

        A record whose file is absent reports ``False`` rather than raising, so a
        partial mirror can be described in one call instead of failing on the
        first gap.
        """
        result: dict[str, bool] = {}
        for record in self.manifest().files:
            target = osp.join(self.dir, record.path)
            if not osp.exists(target):
                result[record.path] = False
                continue
            with open(target, "rb") as handle:
                result[record.path] = _sha256(handle.read()) == record.sha256
        return result

    def ensure_verified(self, asset: StringAsset) -> str:
        """Verify one mirrored asset against its manifest record and return its path.

        Every read path goes through this, so mirrored bytes are never consumed
        on trust. ``BiggGEM`` verifies on each model load for the same reason;
        the hash is cheap next to parsing a million-row links table, and the
        failure it catches is a build that records one file and reads another.
        """
        target = self.path(asset)
        if not osp.exists(target):
            raise FileNotFoundError(
                f"{target} is not mirrored; run the provisioning script for "
                f"taxon {self.taxon} at release {self.release}"
            )
        record = next(
            (r for r in self.manifest().files if r.path == osp.basename(target)), None
        )
        if record is None:
            raise MirrorDriftError(
                f"{self.manifest_path} carries no record for "
                f"{osp.basename(target)}; the file is present but unpinned"
            )
        if self._verified.get(asset) is True:
            return target
        with open(target, "rb") as handle:
            digest = _sha256(handle.read())
        if digest != record.sha256:
            raise MirrorDriftError(
                f"{target} hashes to {digest}, manifest pins {record.sha256}"
            )
        self._verified[asset] = True
        return target

    def strip_prefix(self, protein_id: str) -> str:
        """Drop the ``{taxon}.`` prefix STRING puts on every protein id."""
        prefix = f"{self.taxon}."
        if protein_id.startswith(prefix):
            return protein_id[len(prefix) :]
        return protein_id

    def links_detailed(self) -> pd.DataFrame:
        """Read the detailed links table, with the taxon prefix already stripped."""
        frame = pd.read_csv(
            self.ensure_verified(StringAsset.LINKS_DETAILED),
            sep=" ",
            compression="gzip",
        )
        prefix = f"{self.taxon}."
        for column in ("protein1", "protein2"):
            frame[column] = frame[column].str.removeprefix(prefix)
        return frame

    def preferred_names(self) -> dict[str, str]:
        """Map each stripped protein id to STRING's preferred display name."""
        frame = pd.read_csv(
            self.ensure_verified(StringAsset.INFO), sep="\t", compression="gzip"
        ).rename(columns=lambda name: name.lstrip("#"))
        ids = frame["string_protein_id"].str.removeprefix(f"{self.taxon}.")
        return dict(zip(ids, frame["preferred_name"], strict=True))

    def alias_sources(self) -> dict[str, int]:
        """Every alias source in this taxon's table, with its row count.

        Published so a caller can name a source that exists. The names are not
        stable vocabulary across STRING releases, and a source that is absent is
        the difference between an empty mapping and a wrong one.
        """
        frame = self._aliases_frame()
        counts = frame["source"].value_counts()
        return {str(name): int(count) for name, count in counts.items()}

    def _aliases_frame(self) -> pd.DataFrame:
        """The aliases table with the taxon prefix stripped from the protein id."""
        frame = pd.read_csv(
            self.ensure_verified(StringAsset.ALIASES), sep="\t", compression="gzip"
        ).rename(columns=lambda name: name.lstrip("#"))
        frame["string_protein_id"] = frame["string_protein_id"].str.removeprefix(
            f"{self.taxon}."
        )
        return frame

    def alias_map(self, sources: tuple[str, ...]) -> dict[str, str]:
        """Map each stripped protein id to an alias, from the first matching source.

        ``sources`` is tried in order, which is how a caller states a namespace
        preference. Two things are refused rather than resolved silently, because
        both produce a mapping that looks fine and is wrong:

        * a source name absent from this taxon's table, which would otherwise
          return an empty or partial mapping with no signal. The real names are
          release-specific, so :meth:`alias_sources` publishes them.
        * a protein carrying more than one alias within the chosen source, where
          file order would decide. This is not hypothetical: in the v12.5 slice
          for taxon 160488 every one of the 5,564 proteins has two ``RefSeq``
          aliases, a GenBank protein accession and the locus tag, so asking
          ``RefSeq`` for locus tags silently returned accessions.

        A protein with no alias in any named source is absent from the result
        rather than mapped to itself, so coverage stays visible.
        """
        frame = self._aliases_frame()
        available = set(frame["source"].unique())
        unknown = [name for name in sources if name not in available]
        if unknown:
            raise ValueError(
                f"alias source(s) {unknown} are not in the v{self.release} table "
                f"for taxon {self.taxon}; available sources and their row counts "
                f"are given by alias_sources()"
            )

        mapping: dict[str, str] = {}
        for source in sources:
            subset = frame[frame["source"] == source]
            grouped = subset.groupby("string_protein_id")["alias"].nunique()
            ambiguous = grouped[grouped > 1]
            if len(ambiguous) > 0:
                examples = sorted(ambiguous.index)[:3]
                raise ValueError(
                    f"{len(ambiguous)} protein(s) carry more than one alias in "
                    f"source {source!r} (for example {examples}); file order "
                    f"would decide the mapping. Choose a source with one alias "
                    f"per protein, such as one of the RefSeq_locus / "
                    f"UniProt_GN_OrderedLocusNames style sources."
                )
            for protein_id, alias in zip(
                subset["string_protein_id"], subset["alias"], strict=True
            ):
                mapping.setdefault(protein_id, alias)
        return mapping


def string_channel_graphs(
    mirror: StringTaxonMirror,
    gene_set: GeneSet,
    channels: tuple[str, ...] = STRING_SCORE_CHANNELS,
    above_score: int = 0,
    protein_to_gene: dict[str, str] | None = None,
) -> dict[str, nx.Graph]:
    """Build one undirected graph per evidence channel, restricted to ``gene_set``.

    ``protein_to_gene`` renames STRING protein ids before the gene-set filter.
    It is needed only when the ids are not already the names ``gene_set`` uses;
    for a bacterium whose STRING ids are its locus tags it is left ``None``.

    Edge weight is the channel score, and a node that ends with no edges is not
    added, which is the same shape the yeast path produces.

    ``above_score`` is a STRICT threshold, and is named that way because
    ``min_score=400`` reading as "scores of 400 and up" while excluding the rows
    that score exactly 400 is the kind of off-by-one that never gets noticed. The
    default of 0 keeps every nonzero channel score, which reproduces the yeast
    path exactly.
    """
    unknown = [name for name in channels if name not in STRING_ALL_CHANNELS]
    if unknown:
        raise ValueError(
            f"unknown STRING channel(s): {', '.join(unknown)}. "
            f"Valid channels are: {', '.join(STRING_ALL_CHANNELS)}"
        )

    frame = mirror.links_detailed()
    if protein_to_gene is not None:
        present = set(frame["protein1"]) | set(frame["protein2"])
        unmapped = sorted(present - set(protein_to_gene))
        if unmapped:
            raise ValueError(
                f"protein_to_gene covers {len(present) - len(unmapped)} of "
                f"{len(present)} STRING proteins; {len(unmapped)} are unmapped "
                f"(for example {unmapped[:3]}). Mapping them to nothing would "
                f"drop their edges and return a smaller graph with no signal, so "
                f"supply a complete map or leave protein_to_gene as None and let "
                f"the gene_set filter do the restricting."
            )
        frame["protein1"] = frame["protein1"].map(protein_to_gene)
        frame["protein2"] = frame["protein2"].map(protein_to_gene)

    members = set(gene_set)
    in_set = frame["protein1"].isin(members) & frame["protein2"].isin(members)
    frame = frame[in_set]

    graphs: dict[str, nx.Graph] = {}
    for channel in channels:
        subset = frame.loc[
            frame[channel] > above_score, ["protein1", "protein2", channel]
        ].rename(columns={channel: "weight"})
        graph = nx.from_pandas_edgelist(
            subset,
            source="protein1",
            target="protein2",
            edge_attr="weight",
            create_using=nx.Graph,
        )
        nx.set_edge_attributes(graph, mirror.release, "version")
        graphs[channel] = graph
        log.info(
            "STRING v%s %s: %d nodes, %d edges",
            mirror.release,
            channel,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
    return graphs


def string_graph_name(release: str, channel: str) -> str:
    """The graph name a channel takes, matching the yeast ``string12_0_*`` form."""
    return f"string{release.replace('.', '_')}_{channel}"


def build_string_gene_multigraph(
    mirror: StringTaxonMirror,
    gene_set: GeneSet,
    channels: tuple[str, ...] = STRING_SCORE_CHANNELS,
    above_score: int = 0,
    protein_to_gene: dict[str, str] | None = None,
) -> GeneMultiGraph:
    """Build the per-channel :class:`GeneMultiGraph` for one taxon's STRING slice."""
    channel_graphs = string_channel_graphs(
        mirror,
        gene_set,
        channels=channels,
        above_score=above_score,
        protein_to_gene=protein_to_gene,
    )
    graphs: SortedDict[str, GeneGraph] = SortedDict()
    for channel, graph in channel_graphs.items():
        name = string_graph_name(mirror.release, channel)
        graphs[name] = GeneGraph(name=name, graph=graph, max_gene_set=gene_set)
    return GeneMultiGraph(graphs=graphs)

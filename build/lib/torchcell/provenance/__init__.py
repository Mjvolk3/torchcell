# torchcell/provenance/__init__.py
"""Schema-dependency tracking: flag datasets for rebuild when the schema contract changes.

Two orthogonal, complementary tools -- both driven purely by LOCAL git state, so they work
across machines with no central database:

  schema_impact  (proactive)   diff the working-tree schema vs a git ref, classify each change
                               breaking/stale, and report which dataset loaders must rebuild.
                               Runs as a pre-commit hook to stop a schema edit from silently
                               breaking an unchanged loader.
  build_manifest (retroactive) each built LMDB stores the contract fingerprints of the schema
                               symbols its loader depends on; `check` recomputes them against
                               the local schema and reports which built datasets are now stale.

Fingerprints hash a symbol's CONTRACT (field shape + validators), not its source text, so they
are machine-independent and a benign edit (docstring, method, field description, reorder) flags
nobody. See `torchcell.provenance.schema_deps` for the analysis core.
"""

from torchcell.provenance.build_manifest import (
    BuildManifest,
    DatasetCheck,
    StaleResult,
    SymbolDrift,
    check_all,
    check_manifest,
    compute_manifest,
    write_build_manifest,
)
from torchcell.provenance.schema_deps import (
    ContractSpec,
    FieldSpec,
    SchemaSurface,
    contract_spec,
    fingerprint,
    forward_closure,
    load_default_surface,
    load_surface,
    load_surface_from_sources,
    loader_closure,
    loader_schema_deps,
)
from torchcell.provenance.schema_impact import (
    ChangeKind,
    ImpactReport,
    LoaderImpact,
    SymbolChange,
    build_impact_report,
    classify_change,
    diff_surfaces,
)

__all__ = [
    # schema_deps
    "ContractSpec",
    "FieldSpec",
    "SchemaSurface",
    "contract_spec",
    "fingerprint",
    "forward_closure",
    "load_default_surface",
    "load_surface",
    "load_surface_from_sources",
    "loader_closure",
    "loader_schema_deps",
    # schema_impact
    "ChangeKind",
    "ImpactReport",
    "LoaderImpact",
    "SymbolChange",
    "build_impact_report",
    "classify_change",
    "diff_surfaces",
    # build_manifest
    "BuildManifest",
    "DatasetCheck",
    "StaleResult",
    "SymbolDrift",
    "check_all",
    "check_manifest",
    "compute_manifest",
    "write_build_manifest",
]

---
id: 0cys4ggr77qb7cgsouazowj
title: Private_torchcell
desc: ''
updated: 1791435266450
created: 1791435266450
---

## 2026.10.07 - Visibility design: in-house datasets that never leave the house

In-house wet-lab data is being added as torchcell datasets. It has to be ontologized the
same way every published dataset is, and it must not reach the public knowledge graph or a
`tc-data` release. The design is one enum, one namespace and two gates, each reading the
same fact off the loader class.

### `Visibility`, a ClassVar on the loader

`torchcell/data/experiment_dataset.py`:

```python
class Visibility(StrEnum):
    public = "public"
    private = "private"


class ExperimentDataset(Dataset, ABC):
    visibility: ClassVar[Visibility] = Visibility.public
```

A ClassVar, not a field: visibility is a property of the LOADER, decided once in code, and
both gates read it off the class without instantiating it (instantiating would open an
LMDB, which a gate must not need to do). The default is `public`, so no existing loader
changes behavior and no existing record changes shape.

### The namespace

`torchcell/datasets/private_torchcell/`. Every dataset in it sets
`visibility = Visibility.private`. The package docstring states that contract, because that
is where a loader author will be looking.

### The build gate

`torchcell/knowledge_graphs/dataset_adapter_map.py` now holds TWO maps:

- `dataset_adapter_map`, the public one, unchanged in content.
- `PRIVATE_DATASET_ADAPTER_MAP`, empty for now. A private loader registers its adapter
  HERE, never in the public map, so the public build cannot reach it even by name.

`build_adapter_map(include_private=False)` unions the private map in only when asked, and
`refuse_private_datasets(classes, include_private=False)` raises `PrivateDatasetRefused`
naming every private class and the flag that would permit it. `create_kg.py` and
`create_scerevisiae_kg.py` call it after the config is resolved and BEFORE the first loader
is constructed, so a public build cannot half-produce a store holding in-house records.

Refusing rather than silently dropping is deliberate: a build that quietly omitted a
requested dataset would produce a store that does not match its own config, and nothing
would say so.

The flag is `--include-private`, and `take_include_private_flag(sys.argv)` REMOVES it from
argv before hydra runs, because hydra parses the same argv and fails on an option it does
not know.

`kg_manifest.py` records `visibility` per served dataset (`KgDatasetEntry.visibility`,
defaulting to `"public"`, which is correct for an entry written before the field existed
since a private dataset could never have been served), and `check_admission` blocks a
private dataset outright. There is no flag there on purpose: incremental admission writes
into the SERVED store, which is the public one, so an in-house graph is a different store,
not a flag on this one.

### The release gate

`scripts/package_dataset_lmdb.py` is the only path into the `tc-data` artifact store.
`refuse_if_private` resolves the build manifest's `loader_class` in the dataset registry
and refuses when that class is private, before anything is written. The check is on the
class, not on a path or a config, so there is no spelling of the command that publishes
in-house data. A `loader_class` the registry does not hold is left alone: that is
unregistered, not private, which is a different and pre-existing condition.

### Why filtering is exact rather than heuristic

Every record carries `dataset_name`, which is the dataset class's own name
(`ExperimentDataset.__init__` sets `self.name = self.__class__.__name__` and each loader
writes it into every experiment and reference). So the private records in any store are
exactly the records whose `dataset_name` names a class with `visibility is private`. No
string matching on paths, no per-record flag to keep in sync, and an audit of a store is a
set membership test.

### The source side: `Publication.source_type`

The in-house data's source is a dissertation or a preliminary-exam report with no DOI, so
`Publication` gained a `source_type` (`journal_article` default, plus `dissertation`,
`preliminary_report`, `in_house`) and, for a non-journal source, a required `title` and
`identifier`, where the identifier is the deposited document's mirror-relative path plus
`sha256:<hex>`. A journal article keeps today's rule exactly. Details and the schema-closure
consequence are in [[torchcell.datamodels.schema]].

### Status

The enum, the namespace, both gates and their tests are landed. `PRIVATE_DATASET_ADAPTER_MAP`
is empty: the first private loader and its mirror are being prepared separately.

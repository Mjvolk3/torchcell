---
id: tzhn0fd8necetui51lbgliu
title: Bacterial Graph Classes
desc: ''
updated: 1791377576460
created: 1791377576460
---

## 2026.10.07 - The graph half of the bacterial schema (plan Step 6)

Plan: [[plan.bacteria-ontology-genome]], section 5 items 1 and 3, Step 6. The pydantic half
(the five perturbation leaves, `AssemblyReferenceGenome`, the three phenotypes and their
experiment pairs) is [[torchcell.datamodels.bacterial-perturbation-ontology]]. This step
makes those records queryable in Cypher without changing anything the served store holds.

### Classes added to `biocypher/config/torchcell_schema_config.yaml`

| class | `is_a` | properties | emitted by |
|---|---|---|---|
| `bacterial perturbation` | `genotype` | `systematic_gene_name`, `perturbed_gene_name`, `perturbation_type`, `description`, `gene_namespace` | `bacterial perturbation (chunked)` |
| `flux phenotype` | `phenotypic feature` | envelope + `net_flux`, `net_flux_lower`, `net_flux_upper`, `confidence_level`, `measurement_type`, `n_samples`, `sample_unit`, `target_reaction_ids` | `flux phenotype (chunked)`, `flux phenotype reference` |
| `product titer phenotype` | `phenotypic feature` | envelope + `product`, `titer`, `titer_unit`, `titer_se`, `titer_uncertainty`, `titer_uncertainty_type`, `n_samples`, `sample_unit`, `product_yield`, `product_yield_unit`, `productivity`, `productivity_unit`, `quantification_method` | `product titer phenotype (chunked)`, `product titer phenotype reference` |
| `protein turnover phenotype` | `phenotypic feature` | envelope + `degradation_rate`, `degradation_rate_se`, `half_life`, `synthesis_rate`, `n_replicates`, `measurement_type` | `protein turnover phenotype (chunked)`, `protein turnover phenotype reference` |

The envelope is `graph_level`, `label_name`, `label_statistic_name`. Every `is_a` resolves
in the pinned Biolink 3.2.1 mirror, checked in-process by
`tests/torchcell/knowledge_graphs/test_graph_schema_ontology.py` (BioCypher's own
`OntologyMapping` + `Ontology` on `biocypher/ontology/`, under a second):

- `bacterial perturbation -> genotype -> biological entity -> named thing -> entity`
- each phenotype `-> phenotypic feature -> disease or phenotypic feature -> biological entity -> named thing -> entity`

`bacterial perturbation` is a SIBLING of `perturbation` under `genotype`, not a child. A
child would carry the `Perturbation` label, and every served `MATCH (:Perturbation)` would
start returning bacterial nodes. The test pins that neither class is an ancestor of the
other and that `perturbation`'s ancestry is unchanged.

Three served edge classes gain endpoint labels, and no edge class or edge method is added:

- `perturbation member of`: source `[perturbation, bacterial perturbation]`. A list source
  requires `input_label` in BioCypher (`_horizontal_inheritance_source` reads it), so
  `input_label: perturbation member of` was added, the same shape `genotype member of`
  already has. The adapter's relationship label is unchanged.
- `crispr construct member of`: target `[perturbation, bacterial perturbation]`, because
  `BacterialCrisprInterferencePerturbation` carries the shared `CrisprConstruct`.
- `phenotype member of`: the three phenotype classes join its sources.

### Node id rule and projection

- Id: sha256 of `json.dumps(model_dump())` of the sub-object, the rule every sub-object
  node follows. It is the id `_perturbation_to_genotype_edges`,
  `_crispr_construct_to_perturbation_edges`, `_phenotype_to_experiment_edge` and
  `_get_phenotype_to_experiment_reference_edges` already compute, so those served edge
  methods connect the new nodes unchanged (tested on records built from the Step 4 classes).
- `preferred_id`: `perturbation_type` for `bacterial perturbation` (as `perturbation`);
  `phenotype_<id>` on the experiment side and the class name on the reference side for the
  phenotypes (as `_fitness_phenotype_node`). No `serialized_data` on any of them.
- Phenotype properties are every field except `provenance_gaps`, named by the field so the
  phenotype-to-node-class bijection in `torchcell/datamodels/ontology_checks.py` matches
  them: an enum as its value, a dict as a JSON string, `None` kept as `None`. The pending
  exemption in `test_phenotype_classes_and_node_classes_are_in_bijection` is gone; all
  concrete phenotypes are mapped.
- `_bacterial_perturbation_node` emits only for `BACTERIAL_PERTURBATION_LEAVES` (the five
  leaves, which are exactly the genotype-union members declaring `gene_namespace`, pinned
  by a test). A yeast record emits nothing there.

### Two choices for the owner to confirm before the first bacterial dataset is served

Once a class is served its property set can only change with a full rebuild, so these are
cheap to change now and expensive later.

- **`product` is the JSON of the `Compound`** (name, InChIKey, ChEBI, PubChem, SMILES).
  The bijection check requires every phenotype property to be a field name, so a typed
  `product_inchikey` property would need that check to learn a projection alias first. As
  it stands, a Cypher join on the product's InChIKey is a string match on `product`.
- **No `strain_id` on `bacterial perturbation`.** The plan listed it by analogy with
  `perturbation`, but none of the five leaves declares `strain_id`, so the column would be
  null on every node, which is the "declared but never emitted" failure the coherence check
  exists to catch. The bacterial strain identity lives in leaf fields (`collection` and
  `construction` on the deletion leaf, `barcode` and `library_pool` on the transposon
  leaf), which stay in the experiment record.

### Conf rule for the bacterial adapters (Step 8)

A bacterial adapter conf enables `bacterial perturbation (chunked)` INSTEAD of
`perturbation (chunked)`. The served method emits every leaf of a genotype under the
`perturbation` label, so enabling both writes one content id under two classes and the
import keeps whichever row it reads first.
`test_no_conf_enables_both_perturbation_classes`
(`tests/torchcell/knowledge_graphs/test_adapter_schema_consistency.py`) fails on such a
conf. The rest of the perturbation wiring is the served methods as they are:
`perturbation to genotype (chunked)`, `crispr construct (chunked)`,
`crispr construct to perturbation (chunked)`. A phenotype family enables its `(chunked)`
and `reference` node methods plus `phenotype to experiment (chunked)` and
`phenotype to experiment reference`.

### Deferred: `assembly_set` and `assembly_accession` on the `genome` node

Not added (owner decision, plan section 5 item 2). `genome` is a served class, and a new
property on it reads CHANGED under the admission gate, which is a full rebuild. Nothing is
lost meanwhile: the genome node's id is the sha256 of the full `genome_reference` dump, so
an `AssemblyReferenceGenome` already gets its own node per host, and both fields are in the
node's `serialized_data` and in the experiment reference blob. Deferring costs only Cypher
convenience; the properties can ride the next deliberate full rebuild.

### Admission measurement

`kg_manifest` gained a `drift` subcommand: the graph schema, adapter and value-surface
checks of `check_admission`, run with no dataset named (all 51 mapped datasets are
already served, so `admit` on any of them would read the live store for the superset
proof). It names
CHANGED classes, ADDED classes, served edge classes WIDENED by new endpoint labels, adapter
drift touching served datasets, and ADDED methods; exit 1 when anything served changed.

```bash
python -m torchcell.knowledge_graphs.kg_manifest --manifest <copy of $BUILD_ROOT/database/kg_manifest.json> drift
```

Run against a scratch copy of the served manifest (`/scratch/projects/torchcell/database/kg_manifest.json`,
sha256 `662eaab2...05cd480`, unchanged after the run), store built at `4b293d34`.

Before (the Step 4 merge, schema and adapter as on main):

```text
Served-surface drift  ->  CHANGES SERVED
  graph schema CHANGED: crispr construct
  graph schema ADDED: -
  served edge classes widened (additive): -
  adapter drift touching served datasets: plumbing: _crispr_construct_node_from
  adapter methods ADDED: -
  value surface: unchanged; ADDED: -
```

After:

```text
Served-surface drift  ->  CHANGES SERVED
  graph schema CHANGED: crispr construct
  graph schema ADDED: bacterial perturbation, flux phenotype, product titer phenotype, protein turnover phenotype
  served edge classes widened (additive): crispr construct member of: +bacterial perturbation; perturbation member of: +bacterial perturbation; phenotype member of: +flux phenotype, +product titer phenotype, +protein turnover phenotype
  adapter drift touching served datasets: plumbing: _crispr_construct_node_from
  adapter methods ADDED: _bacterial_perturbation_node, _bacterial_perturbation_node_from, _flux_phenotype_node, _flux_properties, _get_flux_phenotype_reference_nodes, _get_product_titer_phenotype_reference_nodes, _get_protein_turnover_phenotype_reference_nodes, _product_titer_phenotype_node, _product_titer_properties, _protein_turnover_phenotype_node, _protein_turnover_properties
  value surface: unchanged; ADDED: -
```

The difference is ADDED classes, widened edges and ADDED methods only. The one CHANGED class
and the one plumbing drift are in both runs: they come from main `caaf2295f` (the
`crispr construct` property `effector_plasmid_uri` renamed to `effector_plasmid_ref` after
the served build), not from this step. That commit already marks the next KG build as a
full rebuild (`before-next-kg-build`), since five served loaders serialize differently
under it.

Two further checks, both one-off on 2026-10-07:

- BioCypher ontology built from the schema before and after (BioCypher 0.5.43, pinned
  Biolink mirror): 55 extended-schema keys before, 64 after, none removed, and no existing
  key's ancestry changed. The nine added keys are the four classes and five virtual leaves
  (`perturbation.perturbation member of`, `bacterial perturbation.perturbation member of`,
  and one `phenotype member of` leaf per new phenotype).
- BioCypher's own neo4j writer, fed the same yeast `genotype`, `perturbation`,
  `crispr construct`, `fitness phenotype` and `experiment` nodes and the four edges between
  them, wrote byte-identical CSVs under the old and the new schema (18 files,
  `diff -r` empty), with types `PerturbationMemberOf`, `CrisprConstructMemberOf`,
  `PhenotypeMemberOf`. A `bacterial perturbation` node is written with
  `:LABEL` `BacterialPerturbation|BiologicalEntity|Entity|Genotype|NamedThing`, no
  `Perturbation`.

### Yeast output equivalence

`tests/torchcell/adapters/test_bacterial_graph_classes.py` runs one yeast record (an SGA
KanMX deletion with a `strain_id`, a plain KanMX deletion, a CRISPRi perturbation with its
construct) through the chunked methods a yeast CRISPR fitness dataset enables. The three
`perturbation` nodes are written out in full, and eight methods (`genotype`,
`perturbation`, `crispr construct`, `fitness phenotype` and the four edges between them)
are pinned by the sha256 of their canonical output. The expected values were computed by
running origin/main `76933585`'s `CellAdapter` on the same record, and the branch
reproduces all of them; `bacterial perturbation (chunked)` emits nothing for it.

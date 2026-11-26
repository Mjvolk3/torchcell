// isoprenol_titer_048.cql
// [[experiments.048-pputida-isoprenol-cgt]]
//
// Experiment 048 query: the Carruthers 2025 isoprenol titer campaign on
// P. putida KT2440, from KG 4.0. Every served record of the dataset is admitted.
//
// Block shape follows experiments/043-expression-proteome-morphology/queries/
// expression_proteome_morphology_043.cql: a property-shaped RETURN of the
// experiment and its reference, no $gene_set test, since the genome never gates
// a record.
//
// The phenotype is ProductTiterPhenotype at graph_level 'global' with
// label_name 'titer' (ug/mL as released, GC-FID). The perturbations on each
// genotype are the CRISPRi guides (1 to 4 per strain, on PP_ locus tags), the
// five constant pIY670 heterologous pathway genes, and for the Source Data
// panel strains a deletion or a native-copy addition.
//
// 502 experiments: 465 campaign (construct x DBTL cycle) and 37 panel. The DBTL
// cycle is not a field on a served record; it is recovered in the experiment
// from the per-cycle control titer carried by the record's own
// ExperimentReference. See pputida_cgt_data.py.

MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'IsoprenolTiterCarruthers2025Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'global'
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

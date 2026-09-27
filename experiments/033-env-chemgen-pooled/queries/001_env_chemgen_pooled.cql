// experiments/033-env-chemgen-pooled/queries/001_env_chemgen_pooled.cql
// [[experiments.033-env-chemgen-pooled.queries.001_env_chemgen_pooled]]
//
// Every environment-response record of the four chemogenomic datasets kept by the
// experiment 031 planning document (notes-tex/031-unified-representation): Vanacloig 2022,
// the application target; Hillenmeyer 2008 HET; Hoepfner 2014, both ploidy arms; and
// Wildenhain 2015. Hillenmeyer 2008 HOM is left out by the decision recorded there
// ("drop hillenmeyer hom"): its heterozygous sibling already covers 80 percent of its
// genes and it brings one non-exact chemical neighbor above 0.5 to the application panel.
// Adding it back is one more block of the same shape.
//
// WHAT THE QUERY DECIDES, AND WHAT IT LEAVES TO THE BUILD AND THE RUN.
//
// The query decides the record population: a dataset, and every perturbed gene of the
// record in $gene_set. Nothing else. No medium, temperature, dose-basis or compound filter,
// because those axes are what the pooled representation has to absorb, and a record with
// no dosed compound (a Hillenmeyer media swap, a temperature or radiation condition) is a
// measured response to an environment and stays. Which records a training arm reads is a
// read-time selection over the built indices, beside the split and the seed, as in 030.
//
// The gene filter is the only place a record can be lost. Vanacloig genotypes carry four
// perturbations, the queried deletion plus the three drug-efflux deletions of the
// sensitized host (PDR1 YGL013C, PDR3 YBL005W, SNQ2 YDR011W), so a smoke gene set must
// include those three or no Vanacloig record returns. A dataset gene that is not in the
// S288C gene set (a dubious ORF retired since the screen) drops its records; the build
// reports the returned count per dataset beside the served count so that loss is visible.
//
// Served counts (kg_manifest.json, release 2026.09.21-ab6d8c5d): Vanacloig 143,218;
// Hillenmeyer HET 2,698,797; Hoepfner 3,124,319; Wildenhain 428,206; 6,394,540 records.
//
// Records are returned as the serialized_data properties, not as nodes, for the reason
// recorded on the 025 query: hydrated nodes are retained by the driver for the life of
// the result and project past machine memory at this scale.

// EnvChemgenVanacloig2022Dataset (haploid, sensitized host, 41 compounds at IC30)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'EnvChemgenVanacloig2022Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WHERE ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// HetHillenmeyer2008Dataset (heterozygous diploid, one of two copies, 290 compounds)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'HetHillenmeyer2008Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WHERE ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// EnvChemgenHoepfner2014Dataset (diploid; HIP heterozygous and HOP homozygous arms, 148 compounds)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'EnvChemgenHoepfner2014Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WHERE ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// EnvChemgenWildenhain2015Dataset (haploid deletion, 242-gene panel, 5,170 compounds)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'EnvChemgenWildenhain2015Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WHERE ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

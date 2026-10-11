---
id: 1hgbsx08cn7ji8gfrv84w88
title: costanzo2016
desc: ''
updated: 1726857736361
created: 1704437089340
---

## Costanzo2016 Notes on Design

- Much of the design comes from the previous `Datasets` `torchcell/datasets/scerevisiae/costanzo2016_deprecated.py` (retired 2026-10-06 to the graveyard), then we also use the newer notion of capturing all data in a pydantic objects to make the transition to adapters easier.

## Legacy SmfCostanzo2016

```python
@define
class SmfCostanzo2016Dataset:
    root: str = field(default="data/torchcell/smf_costanzo2016")
    url: str = field(
        repr=False,
        default="https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
    )
    raw_dir: str = field(init=False, repr=False)
    processed_dir: str = field(init=False, repr=False)
    data: list[BaseExperiment] = field(init=False, repr=False, factory=list)
    reference: list[FitnessExperimentReference] = field(
        init=False, repr=False, factory=list
    )
    reference_index: ReferenceIndex = field(init=False, repr=False)
    reference_phenotype_std_30 = field(init=False, repr=False)
    reference_phenotype_std_26 = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        if osp.exists(osp.join(self.processed_dir, "dataset.json")):
            self.load()
        else:
            self._download()
            self._extract()
            self._cleanup_after_extract()
            self.data, self.reference = self._process_excel()
            self.data, self.reference = self._remove_duplicates()
            self.reference_index = self.get_reference_index()
            self.save()

    # write a get item method to return a single experiment
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _download(self):
        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            download_url(self.url, self.raw_dir)

    def save(self):
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        save_path = osp.join(self.processed_dir, "dataset.json")
        # Create a dictionary to store the serialized data
        serialized_data = {
            "data": [experiment.model_dump() for experiment in self.data],
            "reference": [ref.model_dump() for ref in self.reference],
            "reference_index": [
                ref_idx.model_dump() for ref_idx in self.reference_index.data
            ],
        }
        with open(save_path, "w") as file:
            json.dump(serialized_data, file, indent=4)

    def load(self):
        load_path = osp.join(self.processed_dir, "dataset.json")
        if not osp.exists(load_path):
            raise FileNotFoundError("Saved dataset not found.")

        with open(load_path, "r") as file:
            serialized_data = json.load(file)

        # Deserialize the data back into the appropriate classes
        self.data = [
            FitnessExperiment.model_validate(exp) for exp in serialized_data["data"]
        ]
        self.reference = [
            FitnessExperimentReference.model_validate(ref)
            for ref in serialized_data["reference"]
        ]
        self.reference_index = ReferenceIndex(
            data=[
                ExperimentReferenceIndex.model_validate(ref_idx)
                for ref_idx in serialized_data["reference_index"]
            ]
        )

    def _extract(self):
        zip_path = osp.join(
            self.raw_dir,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)

    def _cleanup_after_extract(self):
        # We are only keeping the smf data for this dataset
        extracted_folder = osp.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        xlsx_file = osp.join(
            extracted_folder, "strain_ids_and_single_mutant_fitness.xlsx"
        )
        if osp.exists(xlsx_file):
            shutil.move(xlsx_file, self.raw_dir)
        if osp.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
        zip_path = osp.join(
            self.raw_dir,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            os.remove(zip_path)

    def _process_excel(self):
        """
        Process the Excel file and convert each row to Experiment instances for 26°C and 30°C separately.
        """
        xlsx_path = osp.join(self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx")
        df = pd.read_excel(xlsx_path)
        # Process the DataFrame to average rows with 'tsa' or 'tsq'
        df = self._average_tsa_tsq(df)
        # This is an approximate since I cannot find the exact value in the paper
        df["Strain_ID_suffix"] = df["Strain ID"].str.split("_", expand=True)[1]

        # Filter out rows where 'Strain_ID_Part2' contains 'ts' or 'damp'
        filter_condition = ~df["Strain_ID_suffix"].str.contains("ts|damp", na=False)
        df_filtered = df[filter_condition]

        self.reference_phenotype_std_26 = (
            df_filtered["Single mutant fitness (26°) stddev"]
        ).mean()
        self.reference_phenotype_std_30 = (
            df_filtered["Single mutant fitness (30°) stddev"]
        ).mean()
        # Process data for 26°C and 30°C
        df_26 = df_filtered[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (26°)",
                "Single mutant fitness (26°) stddev",
            ]
        ].dropna()
        self._process_temperature_data(df_26, 26)

        df_30 = df_filtered[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (30°)",
                "Single mutant fitness (30°) stddev",
            ]
        ].dropna()

        # This is modifying self.data, and self.reference
        self._process_temperature_data(df_30, 30)

        return self.data, self.reference

    def get_reference_index(self):
        # Serialize references for comparability using model_dump
        serialized_references = [
            json.dumps(ref.model_dump(), sort_keys=True) for ref in self.reference
        ]

        # Identify unique references and their indices
        unique_refs = {}
        for idx, ref_json in enumerate(serialized_references):
            if ref_json not in unique_refs:
                unique_refs[ref_json] = {
                    "indices": [],
                    "model": self.reference[idx],  # Store the Pydantic model
                }
            unique_refs[ref_json]["indices"].append(idx)

        # Create ExperimentReferenceIndex instances
        reference_indices = []
        for ref_info in unique_refs.values():
            bool_array = [i in ref_info["indices"] for i in range(len(self.data))]
            reference_indices.append(
                ExperimentReferenceIndex(reference=ref_info["model"], index=bool_array)
            )

        # Return ReferenceIndex instance
        return ReferenceIndex(data=reference_indices)

    def _average_tsa_tsq(self, df):
        """
        Replace 'tsa' and 'tsq' with 'ts' in the Strain ID and average duplicates.
        """
        # Replace 'tsa' and 'tsq' with 'ts' in Strain ID
        df["Strain ID"] = df["Strain ID"].str.replace("_ts[qa]\d*", "_ts", regex=True)

        # Columns to average
        columns_to_average = [
            "Single mutant fitness (26°)",
            "Single mutant fitness (26°) stddev",
            "Single mutant fitness (30°)",
            "Single mutant fitness (30°) stddev",
        ]

        # Averaging duplicates
        df_avg = (
            df.groupby(["Strain ID", "Systematic gene name", "Allele/Gene name"])[
                columns_to_average
            ]
            .mean()
            .reset_index()
        )

        # Merging averaged values back into the original DataFrame
        df_non_avg = df.drop(columns_to_average, axis=1).drop_duplicates(
            ["Strain ID", "Systematic gene name", "Allele/Gene name"]
        )
        df = pd.merge(
            df_non_avg,
            df_avg,
            on=["Strain ID", "Systematic gene name", "Allele/Gene name"],
        )

        return df

    def _process_temperature_data(self, df, temperature):
        """
        Process DataFrame for a specific temperature and add entries to the dataset.
        """
        for _, row in df.iterrows():
            experiment, ref = self.create_experiment(row, temperature)
            self.data.append(experiment)
            self.reference.append(ref)

    def create_experiment(self, row, temperature):
        """
        Create an Experiment instance from a row of the Excel spreadsheet for a given temperature.
        """
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        # Deal with different types of perturbations
        if "ts" in row["Strain ID"]:
            genotype = InterferenceGenotype(
                perturbation=DampPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                )
            )
        elif "damp" in row["Strain ID"]:
            genotype = InterferenceGenotype(
                perturbation=DampPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                )
            )
        else:
            genotype = DeletionGenotype(
                perturbation=DeletionPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                )
            )
        environment = Environment(
            media=Media(name="YEPD", state="solid"),
            temperature=Temperature(value=temperature),
        )
        reference_environment = environment.model_copy()
        # Phenotype based on temperature
        smf_key = f"Single mutant fitness ({temperature}°)"
        smf_std_key = f"Single mutant fitness ({temperature}°) stddev"
        phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=row[smf_key],
            fitness_std=row[smf_std_key],
        )

        if temperature == 26:
            reference_phenotype_std = self.reference_phenotype_std_26
        elif temperature == 30:
            reference_phenotype_std = self.reference_phenotype_std_30
        reference_phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=1.0,
            fitness_std=reference_phenotype_std,
        )

        reference = FitnessExperimentReference(
            reference_genome=reference_genome,
            reference_environment=reference_environment,
            reference_phenotype=reference_phenotype,
        )

        experiment = FitnessExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )
        return experiment, reference

    def _remove_duplicates(self) -> list[BaseExperiment]:
        """
        Remove duplicate BaseExperiment instances from self.data.
        All fields of the object must match for it to be considered a duplicate.
        """
        unique_data = []
        seen = set()

        for experiment, reference in zip(self.data, self.reference):
            # Serialize the experiment object to a dictionary
            experiment_dict = experiment.model_dump()
            reference_dict = reference.model_dump()

            combined_dict = {**experiment_dict, **reference_dict}
            # Convert dictionary to a JSON string for comparability
            combined_json = json.dumps(combined_dict, sort_keys=True)

            if combined_json not in seen:
                seen.add(combined_json)
                unique_data.append((experiment, reference))

        self.data = [experiment for experiment, reference in unique_data]
        self.reference = [reference for experiment, reference in unique_data]

        return self.data, self.reference

    def df(self) -> pd.DataFrame:
        """
        Create a DataFrame from the list of BaseExperiment instances.
        Each instance is a row in the DataFrame.
        """
        rows = []
        for experiment in self.data:
            # Flatten the structure of each BaseExperiment instance
            row = {
                "species": experiment.experiment_reference_state.reference_genome.species,
                "strain": experiment.experiment_reference_state.reference_genome.strain,
                "media_name": experiment.environment.media.name,
                "media_state": experiment.environment.media.state,
                "temperature": experiment.environment.temperature.value,
                "genotype": experiment.genotype.perturbation.systematic_gene_name,
                "perturbed_gene_name": experiment.genotype.perturbation.perturbed_gene_name,
                "fitness": experiment.phenotype.fitness,
                "fitness_std": experiment.phenotype.fitness_std,
                # Add other fields as needed
            }
            rows.append(row)

        return pd.DataFrame(rows)
```

## Querying Problematic Alleles where Allele Names are Swapped in Query and Array

These alleles have same temperature sensitive alleles but swapped in query and in array.

```python
queried_df = df[(df["Query Strain ID"] == 'YGL113W_tsq1382') | (df["Array Strain ID"] == 'YGL113W_tsa1119')]
duplicates_df = queried_df[queried_df.duplicated('combined_name', keep=False)]
sorted_duplicates_df = duplicates_df.sort_values('combined_name')
temperature_sorted_duplicates_df= sorted_duplicates_df[(sorted_duplicates_df['array_perturbation_type'] == "temperature sensitive") & (sorted_duplicates_df['query_perturbation_type'] == "temperature sensitive")]
```

## We Did Away with the Notion of Duplicate Query-Array Genes

```python
# These are the alleles that show in both query and array
# They have very different fitness values depending on query or array
# Since they cannot be swapped, order matters, and so we remove them
TS_ALLELE_PROBLEMATIC = {
    "srv2-ts",
    "apc2-8",
    "frq1-1",
    "act1-3",
    "sgv1-23",
    "dam1-9",
    "dad1-5005",
    "cdc11-2",
    "msl5-5001",
    "sup35-td",
    "emg1-1",
    "cdc20-1",
    "gus1-5001",
    "nse4-ts2",
    "rpg1-1",
    "mvd1-1296",
    "qri1-5001",
    "prp18-ts",
    "tfc8-5001",
    "taf12-9",
    "rpt2-rf",
    "ipl1-1",
    "duo1-2",
    "med6-ts",
    "rna14-5001",
    "cab5-1",
    "prp4-1",
    "nus1-5001",
    "yju2-5001",
    "tbf1-5001",
    "sec12-4",
    "cet1-15",
    "cdc47-ts",
    "ame1-4",
    "rnt1-ts",
    "sld3-5001",
    "lcb2-16",
    "ret2-1",
    "phs1-1",
    "cdc60-ts",
    "sec39-5001",
    "emg1-5001",
    "sec39-1",
}
```

## 2026.10.01 - Named refusals for unsupported temperatures and blank DMF cells (issue #528)

- Before: SMF and DMF chose the reference noise with an if/elif on 26 and 30 C and no else, so any other temperature (DMF on a `TSA22` row) fell through to `UnboundLocalError`. A blank DMF stddev passed the `is not None` guard as NaN and was stored as a NaN SE typed `sample_sd` (and skipped by the reference mean); a blank DMF value surfaced as the phenotype's `Fitness cannot be NaN`.
- Now: `_reference_std_at` returns the 26 or 30 C noise or raises `UnsupportedTemperatureError` ("Costanzo 2016 strain <id> at <T> C: reference noise is defined only at 26 and 30 C"); DMF `preprocess_raw` raises `BlankDoubleMutantFitnessError` with the blank-row count and the first strain pair before any record or reference noise is computed.
- Record-neutral, measured on the pinned raw files (`b4-costanzo2016/classify_costanzo.py` in the fix scratchpad): 20,705,612 SGA rows, `Arraytype/Temp` only DMA30 14,576,634, TSA26 3,141,356, DMA26 2,937,915, TSA30 49,707; 0 rows at 22 C, 0 blank DMF values, 0 blank DMF stddevs. SMF assigns 26 and 30 C itself (20,484 records, 10,260 at 26 C and 10,224 at 30 C), so no real row reaches the refusal.
- Tests: `test_smf_temperature_other_than_26_or_30_is_refused_by_name`, `test_dmf_row_at_22c_is_refused_by_name_while_dmi_stores_it`, `test_dmf_blank_sd_refuses_the_build_by_name`, `test_dmf_blank_fitness_refuses_the_build_by_name`, `test_dmf_suppressor_damp_and_natmx_classes`.

## 2026.10.10 - L0-L4 verification of the dev stores (#889)

Fitness stores run the existing fitness battery unchanged through its single-pass twin `verify_fitness_dataset_streaming` (L0 validates each record as `FitnessExperiment`; every rule of the eager verifier applies, none skipped). `pair_uniqueness` keys on the record's perturbation SET plus strain ids and `screen_id`, so the double and triple arms needed no host-aware variant (the #889 hypothesis held: 0 duplicates on every Kuzmin arm). Interaction stores run the new gene-interaction battery (`torchcell/verification/gene_interaction.py`): every rule applies; `released_values` compares the stored (score, p-value) multiset to the sha256-pinned release, and `fitness_companion_containment` checks every (strain, environment) key against the companion fitness store.

Release counts equal the stores: 20,705,612 rows across the four Data File S1 tables (SGA_DAmP 1,391,958; SGA_ExE 818,570; SGA_ExN_NxE 5,796,145; SGA_NxN 12,698,939) for DMF and DMI; SMF is the 20,484 (strain, temperature) cells of the 10,859-row spreadsheet that state both a fitness and a stddev. The two 20.7M stores ran under slurm (jobs 3766 and 3767, `scripts/gh_no_verifier_store_sweep.slurm`, 3 h 31 min and 4 h 12 min). `dmi_costanzo2016` reproduces the released (epsilon, P-value) multiset exactly and every one of its keys is a `dmf_costanzo2016` record.

Findings, each left failing and filed: #893 (`smf_costanzo2016`), #930 (`dmf_costanzo2016`), #979 (`dmi_costanzo2016`). All three: the `*-supp1` suppressor-carrying strains are stored as plain deletions (224 SMF records on 102 genes; 292,076 DMF/DMI records carry the split spelling), and 35 released ORFs are retired or renamed in R64-4-1 (74 SMF records, 157,605 DMF/DMI records). DMF also: 2,781 released SDs of exactly 0 (SGA_DAmP 115, SGA_ExE 230, SGA_ExN_NxE 1,356, SGA_NxN 1,080) stored as a sample SD with n = 4.

Tables generated by `experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py` from the reports the sweep wrote (`kg4_pre_build_verification_sweep.py --dataset <Class>`, read-only on `processed/`); count oracles derived by `no_verifier_store_release_counts.py` (results `results/no_verifier_store_release_counts.json`).

`smf_costanzo2016`: **FAIL**, 14 rules, 2 failing (report written 2026-10-10T20:52:08)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 20484 records validated |
| L1 | `count` | PASS | observed 20484, expected 20484 |
| L1 | `pair_uniqueness` | PASS | 20484 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 20484 values checked |
| L2 | `se_nonnegative` | PASS | 20484 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 20484 records |
| L1 | `provenance_gaps` | PASS | 20484 documented provenance gaps over 20484/20484 records; 0 deferred field(s): []; 655488 undeclared None values over 9 carrier fields (top: Compound.inchi x245808, Compound.chebi_id x184356, Compound.smiles x61452, ... |
| L1 | `canonical_gene_names` | **FAIL** | 32 genes carry conflicting common-name spellings (136 records; 0 case-only); 35 systematic names are not the genome's current name; 25 common names resolve to another gene |
| L2 | `uncertainty_sanity` | PASS | 20484 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 143388 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 20484 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 0.994 of 5493 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | **FAIL** | 35 systematic names are absent from the current genome over 74 records: YAR037W x2, YAR040C x2, YAR043C x2, YAR044W x2, YBR075W x2, YBR100W x2, YCL006C x2, YCL012W x2, YCL013W x2, YCL026C x2 |

`dmf_costanzo2016`: **FAIL**, 14 rules, 3 failing (report written 2026-10-11T00:22:40)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 20705612 records validated |
| L1 | `count` | PASS | observed 20705612, expected 20705612 |
| L1 | `pair_uniqueness` | PASS | 20705612 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 20705612 values checked |
| L2 | `se_nonnegative` | PASS | 20705612 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 20705612 records |
| L1 | `provenance_gaps` | PASS | 20705612 documented provenance gaps over 20705612/20705612 records; 0 deferred field(s): []; 662579584 undeclared None values over 9 carrier fields (top: Compound.inchi x248467344, Compound.chebi_id x186350508, Compou... |
| L1 | `canonical_gene_names` | **FAIL** | 32 genes carry conflicting common-name spellings (292076 records; 0 case-only); 35 systematic names are not the genome's current name; 25 common names resolve to another gene |
| L2 | `uncertainty_sanity` | **FAIL** | 2781/20705612 labeled uncertainties are a sample dispersion of exactly 0 (a pseudocount artifact, not perfect precision) |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 144939284 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 20705612 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 0.994 of 5848 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | **FAIL** | 35 systematic names are absent from the current genome over 157605 records: YAR037W x4595, YAR040C x4652, YAR043C x4651, YAR044W x4647, YBR075W x4290, YBR100W x4497, YCL006C x4679, YCL012W x4706, YCL013W x4680, YCL026... |

`dmi_costanzo2016`: **FAIL**, 22 rules, 2 failing (report written 2026-10-11T01:03:48)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 20705612 records validated as GeneInteractionExperiment |
| L0 | `reference_structural` | PASS | 20705612 records validated as GeneInteractionExperimentReference |
| L1 | `count` | PASS | observed 20705612, expected 20705612 |
| L1 | `pair_uniqueness` | PASS | 20705612 unique (strain, environment) records, one each |
| L1 | `interaction_order` | PASS | all 20705612 records perturb 2 distinct genes at graph level 'edge' |
| L2 | `score_finite` | PASS | 20705612 interaction scores are finite numbers |
| L2 | `p_value_in_unit_interval` | PASS | 20705612 stated p-values lie in [0, 1] (0 records state none) |
| L2 | `released_values` | PASS | the 20705612 stored (score, p-value) pairs are the released multiset (every row of the four Data File S1 tables) |
| L3 | `reference_zero` | PASS | every one of the 20705612 reference scores is 0 with no p-value, at the experiment's graph level |
| L3 | `reference_environment` | PASS | every one of the 20705612 references is measured in its experiment's own environment |
| L3 | `signed_unclamped` | PASS | 10797074 aggravating and 9880392 alleviating scores, 28146 at exactly zero |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L1 | `provenance_gaps` | PASS | 20705612 documented provenance gaps over 20705612/20705612 records; 0 deferred field(s): []; 745402032 undeclared None values over 13 carrier fields (top: Compound.inchi x248467344, Compound.chebi_id x186350508, Compo... |
| L1 | `canonical_gene_names` | **FAIL** | 32 genes carry conflicting common-name spellings (292076 records; 0 case-only); 35 systematic names are not the genome's current name; 25 common names resolve to another gene |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 144939284 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 20705612 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 0.994 of 5848 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | **FAIL** | 35 systematic names are absent from the current genome over 157605 records: YAR037W x4595, YAR040C x4652, YAR043C x4651, YAR044W x4647, YBR075W x4290, YBR100W x4497, YCL006C x4679, YCL012W x4706, YCL013W x4680, YCL026... |
| L4 | `fitness_companion_containment` | PASS | every one of the 20705612 (strain, environment) keys is a record of dmf_costanzo2016 |

## 2026.10.11 - Released ORFs that R64-4-1 no longer carries are refused (#893, #930, #979)

- **Finding.** The #889 verifier (PR #980) failed `current_genome_genes` and the not-current half of `canonical_gene_names` on all three stores: 35 systematic names absent from the R64-4-1 annotation, over 74 SMF, 157,605 DMF and 157,605 DMI records.
- **Classification** (measured by [[experiments.036-dataset-fixes-before-kg-build.scripts.yeast_gene_id_currency]], slurm job 3829, against the genomes-tier R64-4-1 GFF and SGD's cached per-locus history, no SGD call): of the 35, 0 are renamed, 28 are ORFs merged into a longer locus (24 genes, 3 blocked reading frames, 1 pseudogene) and 7 are retired (no locus, standard name or alias in R64-4-1: YAR037W, YAR040C, YAR043C, YCL006C, YCL013W, YCL026C, YIL080W). Every merged ORF sits in Data File S1 beside a separate deletion strain of its current locus with a different strain number (`YAR044W_dma74` beside `YAR042W_dma76` and `YAR042W_sn3004`), so storing it under the locus would put two different deletions on one gene. Why SGD retired each of the 7 (for example a dubious-ORF deletion) is in no pinned file and was not measured.
- **Rule** ([[torchcell.datasets.systematic_name_currency]]): a renamed ORF maps to its current name only through a pin quoting SGD's nomenclature record; a merged or retired ORF is refused with its typed reason (`merged_into_locus`, `retired_systematic_name`); any other non-current name stops the build. The pins are re-checked against the injected genome on every build.
- **Loader change.** `SmfCostanzo2016Dataset`, `DmfCostanzo2016Dataset` and `DmiCostanzo2016Dataset` take `genome` (injected by `build_dataset_lmdb` by parameter name) and refuse to build without one. The check runs on the SMF `Systematic gene name` and on the query and array systematic names read off the strain ids, AFTER the reference noise is computed, so the reference stays the release's statistic and no kept record changes. Each build writes `preprocess/systematic_name_ledger.json`.
- **Records per store** (input frame measured by job 3829; confirmed on the rebuilt stores below):

| Store | Records before | Refused (merged + retired) | Records after | Renamed | Names |
|---|---:|---:|---:|---:|---:|
| smf_costanzo2016 | 20,484 | 74 | 20,410 | 0 | 35 |
| dmf_costanzo2016 | 20,705,612 | 157,605 | 20,548,007 | 0 | 35 |
| dmi_costanzo2016 | 20,705,612 | 157,605 | 20,548,007 | 0 | 35 |

  The kept records serialize as before; the LMDB keys are positional, so every record after the first refused row moves to a lower key, and `gene_set.json` loses the 35 names.

- **Not done here.** The `*-supp1` suppressor-carrying strains stored as plain deletions (#893 item 1, the split spellings of `canonical_gene_names`) are an owner decision; the 2,781 released DMF SDs of 0 (#930 item 2) belong to the SD-zero work. One more `canonical_gene_names` mismatch is not a gene-ID currency case and is left failing: Data File S1 names strain `YFL061W` "ddi2", which R64-4-1 gives to YNL335W (YFL061W is DDI3 there, with DDI2 as an alias), over 4,526 DMF rows.
- **Verifier follow-up (#980).** The #889 verifier pins `count` to the release (20,484 / 20,705,612) and `released_values` to the whole released (score, p-value) multiset; refusing rows makes both read short by exactly the ledger's refused rows. Its expected counts and released multiset must exclude the rows naming an ORF in `SGA_SYSTEMATIC_REFUSALS` (the ledger gives the count).

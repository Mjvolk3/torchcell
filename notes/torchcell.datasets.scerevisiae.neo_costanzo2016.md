---
id: m7s25o87y51pr5hgel0ijws
title: Neo_costanzo2016
desc: ''
updated: 1726857736362
created: 1704437089340
---

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

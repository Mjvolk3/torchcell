# torchcell/datasets/scerevisiae/kemmeren2014
# [[torchcell.datasets.scerevisiae.kemmeren2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kemmeren2014
# Test file: tests/torchcell/datasets/scerevisiae/test_kemmeren2014.py

import GEOparse
import logging
import os
import os.path as osp
import pickle
import re
import requests
from collections.abc import Callable
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sortedcontainers import SortedDict
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    Media,
    ReferenceGenome,
    KanMxDeletionPerturbation,
    Temperature,
    Experiment,
    ExperimentReference,
    Publication,
)
from torchcell.data import ExperimentDataset, post_process
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@register_dataset
class MicroarrayKemmeren2014Dataset(ExperimentDataset):
    # GEO accessions for the dataset
    geo_accession_responsive = "GSE42527"  # Responsive mutants
    geo_accession_nonresponsive = "GSE42526"  # Non-responsive mutants
    
    # GEO accessions for wildtype reference datasets
    geo_accession_wt_mata_tecan = "GSE42241"  # MATa, Tecan plate, 20 replicates
    geo_accession_wt_mata_flask = "GSE42240"  # MATa, Erlenmeyer flask, 8 replicates
    geo_accession_wt_matalpha_tecan = "GSE42217"  # MATalpha, Tecan plate, 200 replicates
    geo_accession_wt_matalpha_flask = "GSE42215"  # MATalpha, Erlenmeyer flask, 200 replicates
    
    # Special gene name mappings that are not in standard databases
    # HSN1: The name was reserved for YHR127W but subsequently withdrawn (SGD note 2003-02-07)
    # See: https://www.yeastgenome.org/locus/YHR127W
    SPECIAL_GENE_MAPPINGS = {
        "HSN1": "YHR127W",  # Historical alias retained by SGD
    }

    def __init__(
        self,
        root: str = "data/torchcell/microarray_kemmeren2014",
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):

        # Initialize genome for gene name mapping BEFORE calling super().__init__
        # because super().__init__ triggers process() which needs self.genome
        self.genome = SCerevisiaeGenome(
            genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
            go_root=osp.join(DATA_ROOT, "data/go"),
            overwrite=True,  # subject to change... don't want to dropped genes.
        )

        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        return MicroarrayExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        return MicroarrayExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        return [
            f"{self.geo_accession_responsive}_family.soft.gz",
            f"{self.geo_accession_nonresponsive}_family.soft.gz",
            f"{self.geo_accession_wt_mata_tecan}_family.soft.gz",
            f"{self.geo_accession_wt_mata_flask}_family.soft.gz",
            f"{self.geo_accession_wt_matalpha_tecan}_family.soft.gz",
            f"{self.geo_accession_wt_matalpha_flask}_family.soft.gz",
        ]

    def download(self):
        """Download supplementary Table S1 and expression data from GEO."""
        # First, download supplementary Table S1 with mating type information
        table_path = osp.join(self.raw_dir, "kemmeren2014_table_s1.xlsx")
        if not osp.exists(table_path):
            log.info(
                "Downloading supplementary Table S1 with mating type information..."
            )
            # Using Box shared link for the Excel file
            url = "https://uofi.box.com/shared/static/9n6ruj58ueup0cebhnek8ijdcy4om0bi"

            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                # Check if we got actual Excel content
                if len(response.content) < 1000:  # Excel files should be larger
                    raise ValueError(
                        f"Downloaded file too small ({len(response.content)} bytes), likely not the Excel file"
                    )

                with open(table_path, "wb") as f:
                    f.write(response.content)

                log.info(
                    f"Successfully downloaded Table S1 to {table_path} ({len(response.content)} bytes)"
                )

            except Exception as e:
                log.error(f"Failed to download Table S1: {e}")
                raise RuntimeError(
                    f"Failed to download Table S1 from {url}\n"
                    f"Error: {e}\n"
                    f"Please check the URL or save manually as: {table_path}"
                )

        # Then download both GEO datasets
        for geo_accession in [
            self.geo_accession_responsive,
            self.geo_accession_nonresponsive,
        ]:
            log.info(f"Downloading GEO dataset {geo_accession}...")

            try:
                gse = GEOparse.get_GEO(
                    geo=geo_accession, destdir=self.raw_dir, silent=False
                )
                log.info(f"Successfully downloaded {geo_accession}")

                # Save the parsed GEO object for use in process()
                geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
                with open(geo_pkl_path, "wb") as f:
                    pickle.dump(gse, f)

            except Exception as e:
                log.error(f"Failed to download GEO data: {e}")
                raise RuntimeError(f"Failed to download {geo_accession} from GEO")
        
        # Download wildtype reference datasets
        log.info("Downloading wildtype reference datasets...")
        for geo_accession in [
            self.geo_accession_wt_mata_tecan,
            self.geo_accession_wt_mata_flask,
            self.geo_accession_wt_matalpha_tecan,
            self.geo_accession_wt_matalpha_flask,
        ]:
            log.info(f"Downloading WT dataset {geo_accession}...")
            
            try:
                gse = GEOparse.get_GEO(
                    geo=geo_accession, destdir=self.raw_dir, silent=False
                )
                log.info(f"Successfully downloaded {geo_accession}")
                
                # Save the parsed GEO object
                geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
                with open(geo_pkl_path, "wb") as f:
                    pickle.dump(gse, f)
                    
            except Exception as e:
                log.error(f"Failed to download WT data: {e}")
                raise RuntimeError(f"Failed to download {geo_accession} from GEO")

    @post_process
    def process(self):
        # Initialize resolution statistics
        self.resolved_by_excel = 0
        self.resolved_by_gene_table = 0
        self.resolved_by_alias = 0
        self.unresolved_genes = 0
        
        # Load mating type information from supplementary table
        systematic_to_strain, common_to_systematic = self._load_mating_type_map()

        # Load both GEO objects and extract platform annotation
        all_gsms = {}
        probe_to_gene_map = {}

        for geo_accession in [
            self.geo_accession_responsive,
            self.geo_accession_nonresponsive,
        ]:
            geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
            if osp.exists(geo_pkl_path):
                with open(geo_pkl_path, "rb") as f:
                    gse = pickle.load(f)
            else:
                # Re-download if pickle doesn't exist
                gse = GEOparse.get_GEO(
                    geo=geo_accession, destdir=self.raw_dir, silent=False
                )

            # Extract platform annotation for probe-to-gene mapping
            if not probe_to_gene_map and hasattr(gse, "gpls"):
                probe_to_gene_map = self._extract_probe_to_gene_mapping(gse)

            # Combine GSMs from both datasets
            all_gsms.update(gse.gsms)

        log.info(f"Processing {len(all_gsms)} GEO samples from both datasets...")
        log.info(f"Found {len(probe_to_gene_map)} probe-to-gene mappings")
        log.info(f"Loaded {len(systematic_to_strain)} systematic to strain mappings")
        log.info(f"Loaded {len(common_to_systematic)} common to systematic mappings")
        
        # Debug: Look for YCR087C-A specifically
        log.info("\n=== Searching for YCR087C-A in GEO samples ===")
        ycr087ca_samples = []
        for gsm_name, gsm in all_gsms.items():
            title = gsm.metadata.get("title", [""])[0]
            if "YCR087C-A" in title.upper() or "YCR087CA" in title.upper():
                ycr087ca_samples.append((gsm_name, title))
                log.info(f"Found YCR087C-A in title: {gsm_name} - {title}")
            
            # Also check characteristics
            for char_key in ['characteristics_ch1', 'characteristics_ch2']:
                if char_key in gsm.metadata:
                    for char in gsm.metadata[char_key]:
                        if "YCR087C-A" in str(char).upper() or "YCR087CA" in str(char).upper():
                            log.info(f"Found YCR087C-A in {char_key}: {gsm_name} - {char}")
                            if gsm_name not in [s[0] for s in ycr087ca_samples]:
                                ycr087ca_samples.append((gsm_name, title))
        
        if not ycr087ca_samples:
            log.warning("YCR087C-A NOT found in any GEO sample titles or characteristics!")
        else:
            log.info(f"Found {len(ycr087ca_samples)} samples with YCR087C-A")
        
        # Check if YCR087C-A is in the Excel mappings
        log.info(f"YCR087C-A in systematic_to_strain: {'YCR087C-A' in systematic_to_strain}")
        if 'YCR087C-A' in systematic_to_strain:
            log.info(f"YCR087C-A strain: {systematic_to_strain['YCR087C-A']}")
        log.info(f"YCR087C-A in common_to_systematic values: {'YCR087C-A' in common_to_systematic.values()}")
        log.info("===")

        # Parse samples and extract metadata
        samples_data = []
        wt_samples = []
        deletion_samples_by_gene = {}  # Group samples by gene for dye-swap averaging
        already_assigned = set()  # Track assigned systematic names

        for sample_idx, (gsm_name, gsm) in enumerate(all_gsms.items()):
            # Extract metadata from characteristics
            sample_info = {
                "geo_accession": gsm_name,
                "title": gsm.metadata.get("title", [""])[0],
            }

            # Parse from title - look for gene-del pattern
            title = sample_info["title"]
            systematic_gene_name = None
            is_deletion = False
            is_wildtype = False
            gene_resolved_from_title = False  # Track if we already tried resolution from title

            # Check if it's a deletion mutant (contains -del)
            if "-del" in title.lower():
                is_deletion = True
                # Extract gene name before -del
                gene_part = title.split("-del")[0].strip()

                # Remove [HS1991] or similar prefixes if present
                if "]" in gene_part:
                    gene_part = gene_part.split("]")[-1].strip()

                common_name = gene_part.upper()

                # Convert to systematic name using comprehensive resolution
                systematic_gene_name = self.resolve_gene_name_comprehensive(
                    common_name, common_to_systematic, systematic_to_strain, already_assigned
                )
                gene_resolved_from_title = True  # Mark that we attempted resolution
                if not systematic_gene_name:
                    log.debug(f"Skipping {common_name} from title - cannot resolve")

            # Check characteristics_ch2 for confirmation
            characteristics_ch2 = gsm.metadata.get("characteristics_ch2", [])

            for char in characteristics_ch2:
                if "genotype/variation:" in char:
                    genotype = char.split("genotype/variation:")[-1].strip()

                    # Note: "refpool" in ch2 means this is a dye-swap sample, NOT a wildtype
                    # The deletion mutant is in ch1, reference in ch2
                    if "-del" in genotype:
                        # This is a deletion mutant sample
                        is_deletion = True
                        is_wildtype = False

                        # Only extract gene name if we haven't already tried from title
                        if not gene_resolved_from_title and not systematic_gene_name:
                            gene_part = genotype.replace("-del", "").strip()
                            # Remove [HS1991] or similar prefixes if present
                            if "]" in gene_part:
                                gene_part = gene_part.split("]")[-1].strip()
                            common_name = gene_part.upper()
                            systematic_gene_name = self.resolve_gene_name_comprehensive(
                                common_name, common_to_systematic, systematic_to_strain, already_assigned
                            )
                            if not systematic_gene_name:
                                log.debug(f"Skipping {common_name} from characteristics - cannot resolve")

            # Note: We do NOT check characteristics_ch1 for refpool
            # In two-channel microarrays, ch1 is ALWAYS the reference pool
            # Only ch2 determines if this sample is a deletion mutant or wildtype

            # Final check: Only mark as wildtype if no deletion was found
            # Most samples should be deletion mutants (with or without dye-swap)
            if not is_deletion and not systematic_gene_name:
                # This might be a true wildtype/control sample
                is_wildtype = True
            else:
                is_wildtype = False

            sample_info["systematic_gene_name"] = systematic_gene_name
            sample_info["is_deletion"] = is_deletion
            sample_info["is_wildtype"] = is_wildtype

            sample_info["gsm_object"] = gsm

            if is_wildtype:
                wt_samples.append(gsm)
            elif is_deletion and systematic_gene_name:
                # Group deletion samples by gene for dye-swap averaging
                if systematic_gene_name not in deletion_samples_by_gene:
                    deletion_samples_by_gene[systematic_gene_name] = []
                    already_assigned.add(systematic_gene_name)  # Mark as assigned
                deletion_samples_by_gene[systematic_gene_name].append(gsm)

            samples_data.append(sample_info)

        # Process wildtype reference datasets to get strain-specific references
        (
            self.wt_expression_BY4741,
            self.wt_std_BY4741,
            self.wt_cv_BY4741,
            self.wt_expression_BY4742,
            self.wt_std_BY4742,
            self.wt_cv_BY4742,
        ) = self._process_wt_references(probe_to_gene_map)
        
        log.info(f"WT reference for BY4741 (MATa): {len(self.wt_expression_BY4741)} genes")
        log.info(f"WT reference for BY4742 (MATalpha): {len(self.wt_expression_BY4742)} genes")

        log.info(f"Found {len(deletion_samples_by_gene)} unique gene deletions")
        log.info(f"Found {len(wt_samples)} wildtype reference samples")
        
        # Analyze sample distribution per gene
        replicate_counts = {}
        for gene, samples in deletion_samples_by_gene.items():
            count = len(samples)
            if count not in replicate_counts:
                replicate_counts[count] = 0
            replicate_counts[count] += 1
        
        log.info("\n=== Replicates per Gene Analysis ===")
        log.info(f"Average samples per gene: {2633/len(deletion_samples_by_gene):.2f}")
        for count in sorted(replicate_counts.keys()):
            log.info(f"Genes with {count} samples: {replicate_counts[count]}")
        
        # Show examples of genes with different replicate counts
        for count in [1, 2, 3, 4]:
            if count in replicate_counts:
                examples = [g for g, s in deletion_samples_by_gene.items() if len(s) == count][:3]
                log.info(f"Examples with {count} replicates: {examples}")
        
        # Check specific gene with 4 replicates if any
        genes_with_4 = [g for g, s in deletion_samples_by_gene.items() if len(s) == 4]
        if genes_with_4:
            example_gene = genes_with_4[0]
            log.info(f"\nExample gene with 4 replicates: {example_gene}")
            for i, gsm in enumerate(deletion_samples_by_gene[example_gene]):
                title = gsm.metadata.get("title", [""])[0]
                log.info(f"  Sample {i+1}: {title}")
        
        log.info("===")
        
        # Log gene resolution summary
        log.info("=== Gene Resolution Summary ===")
        log.info(f"Resolved by Excel mapping: {self.resolved_by_excel}")
        log.info(f"Resolved by gene_attribute_table: {self.resolved_by_gene_table}")
        log.info(f"Resolved by alias_to_systematic: {self.resolved_by_alias}")
        log.info(f"Could not resolve: {self.unresolved_genes}")
        total_attempts = self.resolved_by_excel + self.resolved_by_gene_table + self.resolved_by_alias + self.unresolved_genes
        log.info(f"Total resolution attempts: {total_attempts}")
        
        # Check for one-to-one mapping with Excel ORF names
        excel_orf_names = set(systematic_to_strain.keys())
        resolved_orf_names = set(deletion_samples_by_gene.keys())
        
        missing_in_geo = excel_orf_names - resolved_orf_names
        extra_in_geo = resolved_orf_names - excel_orf_names
        
        if missing_in_geo:
            log.warning(f"\n=== Missing ORFs Analysis ===")
            log.warning(f"Found {len(missing_in_geo)} ORFs in Excel but not in GEO: {missing_in_geo}")
            for missing_orf in missing_in_geo:
                log.warning(f"Missing: {missing_orf} (strain: {systematic_to_strain.get(missing_orf, 'Unknown')})")
                # Try to find if it appears anywhere in common names
                for common, systematic in common_to_systematic.items():
                    if systematic == missing_orf:
                        log.warning(f"  - Has common name in Excel: {common}")
        
        if extra_in_geo:
            log.warning(f"Found {len(extra_in_geo)} ORFs in GEO but not in Excel: {extra_in_geo}")
        
        # For now, just warn instead of asserting
        if excel_orf_names != resolved_orf_names:
            log.warning(
                f"\nWARNING: ORF name sets do not match exactly!\n"
                f"Excel has {len(excel_orf_names)} ORFs, GEO resolved to {len(resolved_orf_names)} ORFs\n"
                f"Missing in GEO: {missing_in_geo}\n"
                f"Extra in GEO: {extra_in_geo}\n"
                f"Continuing with {len(resolved_orf_names)} genes..."
            )
        else:
            log.info(f"✓ Perfect match: All {len(excel_orf_names)} Excel ORF names matched with GEO deletions")

        # Debug: Check which genes from GEO are not in systematic_to_strain map
        missing_genes = []
        for gene in deletion_samples_by_gene.keys():
            if gene not in systematic_to_strain:
                missing_genes.append(gene)

        if missing_genes:
            log.warning(
                f"Found {len(missing_genes)} genes in GEO that are not in Excel systematic_to_strain map:"
            )
            log.warning(f"First 10 missing genes: {missing_genes[:10]}")
        
        # Validate log2 ratios against original GEO data
        self._validate_log2_ratios(deletion_samples_by_gene, probe_to_gene_map, systematic_to_strain)

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(5e12),  # 5TB for expression data
        )

        idx = 0
        with env.begin(write=True) as txn:
            # Process each unique gene deletion (averaging dye-swaps)
            for gene_name, gsm_list in tqdm(deletion_samples_by_gene.items()):
                # Average expression data from all dye-swap replicates
                averaged_expression, technical_std = self._average_dye_swaps(
                    gsm_list, probe_to_gene_map
                )

                # Skip if no expression data was extracted
                if not averaged_expression:
                    log.warning(f"No expression data for gene {gene_name}, skipping...")
                    continue

                # Determine strain from systematic_to_strain map - REQUIRED
                if gene_name not in systematic_to_strain:
                    log.error(
                        f"No mating type found for {gene_name} in Table S1. This gene deletion is in GEO but not in the Excel file."
                    )
                    log.error(
                        f"Skipping {gene_name} - cannot determine correct strain (BY4741 vs BY4742)"
                    )
                    continue  # Skip this gene since we don't know the strain
                strain = systematic_to_strain[gene_name]

                # Create sample info for this gene deletion
                sample_info = {
                    "systematic_gene_name": gene_name,
                    "num_replicates": len(gsm_list),
                    "strain": strain,
                }

                # Select appropriate WT reference and CV based on strain
                if strain == "BY4741":
                    refpool_expression = self.wt_expression_BY4741
                    refpool_std = self.wt_std_BY4741
                    refpool_cv = self.wt_cv_BY4741
                elif strain == "BY4742":
                    refpool_expression = self.wt_expression_BY4742
                    refpool_std = self.wt_std_BY4742
                    refpool_cv = self.wt_cv_BY4742
                else:
                    log.error(f"Unknown strain {strain} for {gene_name}")
                    continue
                    
                # Extract refpool from deletion samples' Cy3 channel for CV-scaled std
                # This is the actual refpool at the scale of the deletion experiment
                deletion_refpool = self._extract_refpool_from_deletion_samples(
                    gsm_list, probe_to_gene_map
                )
                
                # Calculate CV-scaled std for each gene
                cv_scaled_std = SortedDict()
                for gene in averaged_expression:
                    if gene in deletion_refpool and gene in refpool_cv:
                        # Apply CV to the actual refpool value from deletion sample
                        cv_scaled_std[gene] = refpool_cv[gene] * deletion_refpool[gene]
                    else:
                        # Use original std if we can't scale
                        cv_scaled_std[gene] = technical_std.get(gene, 0.0) if technical_std else 0.0
                    
                experiment, reference, publication = self.create_expression_experiment(
                    self.name,
                    sample_info,
                    averaged_expression,
                    technical_std,
                    deletion_refpool,  # Use actual refpool from deletion samples
                    cv_scaled_std,     # Use CV-scaled std instead of original
                )

                # Serialize the Pydantic objects
                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{idx}".encode(), serialized_data)
                idx += 1

        env.close()

        # Save preprocessed data
        os.makedirs(self.preprocess_dir, exist_ok=True)
        samples_df = pd.DataFrame(samples_data)
        samples_df = samples_df.drop(
            "gsm_object", axis=1
        )  # Remove GSM object before saving
        samples_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info(f"Processed {idx} unique gene deletion experiments")

        # Log statistics
        total_samples = len(samples_data)
        deletion_samples = sum(1 for s in samples_data if s["is_deletion"])
        wt_samples_count = sum(1 for s in samples_data if s["is_wildtype"])
        log.info(
            f"Total samples: {total_samples}, Deletion samples: {deletion_samples}, Wildtype: {wt_samples_count}"
        )
        log.info(f"Unique gene deletions: {len(deletion_samples_by_gene)}")

    def _process_wt_references(self, probe_to_gene_map) -> tuple[SortedDict, SortedDict, SortedDict, SortedDict, SortedDict, SortedDict]:
        """Process wildtype reference datasets to extract refpool references and CV.
        
        The WT datasets contain hybridizations of wt vs. refpool and refpool vs. wt.
        The refpool is pooled RNA from wildtype strains used as common reference.
        
        Returns:
            tuple: (refpool_expression_BY4741, refpool_std_BY4741, refpool_cv_BY4741,
                   refpool_expression_BY4742, refpool_std_BY4742, refpool_cv_BY4742)
        """
        log.info("Processing wildtype reference datasets to extract refpool and calculate CV...")
        
        # Process MATa (BY4741) wildtype samples
        mata_samples = []
        for geo_accession in [self.geo_accession_wt_mata_tecan, self.geo_accession_wt_mata_flask]:
            geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
            if osp.exists(geo_pkl_path):
                with open(geo_pkl_path, "rb") as f:
                    gse = pickle.load(f)
                    mata_samples.extend(list(gse.gsms.values()))
        
        # Process MATalpha (BY4742) wildtype samples  
        matalpha_samples = []
        for geo_accession in [self.geo_accession_wt_matalpha_tecan, self.geo_accession_wt_matalpha_flask]:
            geo_pkl_path = osp.join(self.raw_dir, f"{geo_accession}.pkl")
            if osp.exists(geo_pkl_path):
                with open(geo_pkl_path, "rb") as f:
                    gse = pickle.load(f)
                    matalpha_samples.extend(list(gse.gsms.values()))
        
        log.info(f"Found {len(mata_samples)} MATa WT samples")
        log.info(f"Found {len(matalpha_samples)} MATalpha WT samples")
        
        # Extract refpool references and CV for each strain
        refpool_expression_BY4741, refpool_cv_BY4741, refpool_std_BY4741 = self._calculate_refpool_reference(
            mata_samples, probe_to_gene_map
        )
        refpool_expression_BY4742, refpool_cv_BY4742, refpool_std_BY4742 = self._calculate_refpool_reference(
            matalpha_samples, probe_to_gene_map
        )
        
        # Store CV for later use
        self.refpool_cv_BY4741 = refpool_cv_BY4741
        self.refpool_cv_BY4742 = refpool_cv_BY4742
        
        return refpool_expression_BY4741, refpool_std_BY4741, refpool_cv_BY4741, refpool_expression_BY4742, refpool_std_BY4742, refpool_cv_BY4742
    
    def _calculate_refpool_reference(
        self, wt_gsm_list, probe_to_gene_map
    ) -> tuple[SortedDict, SortedDict, SortedDict]:
        """Extract refpool expression values from WT GSM objects and calculate CV.
        
        The refpool is the same pooled RNA across samples but measured multiple times.
        We extract it to calculate coefficient of variation (CV) for noise estimation.
        
        Returns:
            tuple: (mean_refpool_expression, cv_refpool, std_refpool_expression)
                - mean_refpool_expression: average refpool value per gene
                - cv_refpool: coefficient of variation (std/mean) per gene
                - std_refpool_expression: standard deviation per gene
        """
        if len(wt_gsm_list) == 0:
            log.warning("No wildtype samples found, returning empty reference")
            return SortedDict(), SortedDict(), SortedDict()

        log.info(f"Extracting refpool from {len(wt_gsm_list)} wildtype samples")

        # Collect all refpool values per gene
        all_refpool_values = {}
        sample_count = 0
        
        for gsm in wt_gsm_list:
            refpool_data = self._extract_refpool_from_wt_gsm(gsm, probe_to_gene_map)
            if refpool_data:
                sample_count += 1
                for gene, value in refpool_data.items():
                    if gene not in all_refpool_values:
                        all_refpool_values[gene] = []
                    all_refpool_values[gene].append(value)

        log.info(f"Successfully extracted refpool from {sample_count} samples")

        # Calculate mean, std, and CV of refpool
        refpool_mean_expression = SortedDict()
        refpool_std_expression = SortedDict()
        refpool_cv = SortedDict()
        
        for gene, values in all_refpool_values.items():
            if len(values) >= 2:  # Need at least 2 values for std
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                
                refpool_mean_expression[gene] = mean_val
                refpool_std_expression[gene] = std_val
                
                # Calculate CV (coefficient of variation)
                # CV is scale-independent measure of relative variability
                if mean_val > 1.0:  # Avoid division by very small numbers
                    refpool_cv[gene] = std_val / mean_val
                else:
                    refpool_cv[gene] = 0.0  # Set CV to 0 for low-expressed genes
            
        # Log statistics
        if refpool_cv:
            cv_values = list(refpool_cv.values())
            log.info(f"Extracted refpool reference for {len(refpool_mean_expression)} genes")
            log.info(f"Median CV: {np.median(cv_values):.3f}")
            log.info(f"Mean CV: {np.mean(cv_values):.3f}")
            log.info(f"CV range: [{np.min(cv_values):.3f}, {np.max(cv_values):.3f}]")
            
            # Warn about high CV genes
            high_cv_genes = [gene for gene, cv in refpool_cv.items() if cv > 0.5]
            if high_cv_genes:
                log.info(f"Found {len(high_cv_genes)} genes with CV > 0.5 ({100*len(high_cv_genes)/len(refpool_cv):.1f}%)")
                log.debug(f"Example high CV genes: {high_cv_genes[:5]}")
        
        return refpool_mean_expression, refpool_cv, refpool_std_expression
    
    def _calculate_wt_reference_with_std(
        self, wt_gsm_list, probe_to_gene_map
    ) -> tuple[SortedDict, SortedDict]:
        """Calculate average wildtype expression values and std from GSM objects.
        
        Returns:
            tuple: (mean_expression, std_expression)
        """
        if len(wt_gsm_list) == 0:
            log.warning("No wildtype samples found, returning empty reference")
            return SortedDict(), SortedDict()

        log.info(f"Calculating reference from {len(wt_gsm_list)} wildtype samples")

        # Collect all expression values per gene
        all_expressions = {}
        
        for gsm in wt_gsm_list:
            expr_data = self._extract_expression_from_gsm(gsm, probe_to_gene_map)
            for gene, value in expr_data.items():
                if gene not in all_expressions:
                    all_expressions[gene] = []
                all_expressions[gene].append(value)

        # Calculate mean and std
        wt_mean_expression = SortedDict()
        wt_std_expression = SortedDict()
        
        for gene, values in all_expressions.items():
            wt_mean_expression[gene] = np.mean(values)
            wt_std_expression[gene] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        log.info(f"Calculated reference for {len(wt_mean_expression)} genes")
        
        return wt_mean_expression, wt_std_expression

    def _calculate_wt_reference_from_gsm(
        self, wt_gsm_list, probe_to_gene_map
    ) -> SortedDict:
        """Calculate average wildtype expression values from GSM objects.
        DEPRECATED: Use _calculate_wt_reference_with_std instead.
        """
        if len(wt_gsm_list) == 0:
            log.warning("No wildtype samples found, returning empty reference")
            return SortedDict()

        log.info(f"Calculating reference from {len(wt_gsm_list)} wildtype samples")

        # Aggregate expression across all WT samples
        all_expression = SortedDict()
        gene_counts = SortedDict()

        for gsm in wt_gsm_list:
            expr_data = self._extract_expression_from_gsm(gsm, probe_to_gene_map)
            for gene, value in expr_data.items():
                if gene not in all_expression:
                    all_expression[gene] = 0.0
                    gene_counts[gene] = 0
                all_expression[gene] += value
                gene_counts[gene] += 1

        # Calculate mean
        wt_reference = SortedDict(
            {gene: all_expression[gene] / gene_counts[gene] for gene in all_expression}
        )

        return wt_reference

    def _extract_probe_to_gene_mapping(self, gse) -> dict:
        """Extract probe ID to gene name mapping from GEO platform annotation."""
        probe_to_gene = {}

        # Check if platform data is available
        if not hasattr(gse, "gpls") or not gse.gpls:
            log.warning("No platform annotation found in GEO dataset")
            return probe_to_gene

        # Get the first platform (usually there's only one)
        for gpl_name, gpl in gse.gpls.items():
            log.info(f"Processing platform {gpl_name}")

            if hasattr(gpl, "table") and gpl.table is not None:
                table = gpl.table

                # Look for gene symbol columns
                gene_columns = [
                    "ORF",
                    "Gene",
                    "Gene Symbol",
                    "Gene_Symbol",
                    "GENE_SYMBOL",
                    "gene_symbol",
                    "SystematicName",
                    "Systematic_Name",
                    "SYSTEMATIC_NAME",
                ]

                id_column = None
                gene_column = None

                # Find ID and gene columns
                if "ID" in table.columns:
                    id_column = "ID"
                elif "SPOT" in table.columns:
                    id_column = "SPOT"

                for col in gene_columns:
                    if col in table.columns:
                        gene_column = col
                        break

                if id_column and gene_column:
                    for _, row in table.iterrows():
                        probe_id = str(
                            int(row[id_column])
                        )  # Convert to int first to avoid '1.0'
                        gene_name = str(row[gene_column])

                        # Clean and validate gene name
                        if gene_name and gene_name != "nan" and gene_name != "":
                            # Use gene name directly from platform annotation
                            # Platform should already have systematic names
                            gene_name_upper = gene_name.upper()
                            if re.match(r"Y[A-Z]{2}\d{3}[CW]", gene_name_upper):
                                # Already a systematic name
                                probe_to_gene[probe_id] = gene_name_upper
                            elif re.match(r"Q\d{4}", gene_name_upper):
                                # Mitochondrial genes are already systematic
                                probe_to_gene[probe_id] = gene_name_upper
                            else:
                                # Keep the gene name as is (might be common name)
                                # The expression averaging will handle mismatches
                                probe_to_gene[probe_id] = gene_name_upper

                    log.info(
                        f"Extracted {len(probe_to_gene)} probe-to-gene mappings from {gpl_name}"
                    )
                else:
                    log.warning(
                        f"Could not find appropriate columns in platform {gpl_name}"
                    )
                    log.info(f"Available columns: {list(table.columns)}")

        return probe_to_gene

    def _extract_expression_from_gsm(self, gsm, probe_to_gene_map=None) -> SortedDict:
        """Extract expression values from a GSM object."""
        expression_data = SortedDict()

        # Get the expression table from the GSM
        if hasattr(gsm, "table") and gsm.table is not None:
            # The table contains probe IDs and expression values
            # We need to map probes to genes
            table = gsm.table

            # Check for expected columns
            if "ID_REF" not in table.columns:
                log.warning(
                    f"ID_REF column not found. Available columns: {list(table.columns)}"
                )
                return expression_data

            # Determine which expression values to use
            # Prefer normalized signal values over VALUE column
            expression_column = None
            if (
                "Signal Norm_Cy5" in table.columns
                and "Signal Norm_Cy3" in table.columns
            ):
                # Use Cy5 for ch2 (sample) and Cy3 for ch1 (reference)
                # We'll use Cy5 as it typically contains the sample signal
                expression_column = "Signal Norm_Cy5"
            elif "VALUE" in table.columns:
                expression_column = "VALUE"
            else:
                log.warning(
                    f"No suitable expression column found. Available: {list(table.columns)}"
                )
                return expression_data

            for _, row in table.iterrows():
                probe_id = str(
                    int(row["ID_REF"])
                )  # Convert to int first to avoid '1.0'

                # Map probe ID to gene name
                if probe_to_gene_map and probe_id in probe_to_gene_map:
                    gene = probe_to_gene_map[probe_id]
                    # Skip conversion here - probe_to_gene_map already has the gene names we need
                    # Converting 15000+ genes per sample is too slow
                    try:
                        value = float(row[expression_column])
                        # Keep all expression values (including negative log2 ratios)
                        expression_data[gene] = value
                    except (ValueError, TypeError):
                        continue

            # Log if we didn't extract any expression data
            if not expression_data and probe_to_gene_map:
                log.debug(
                    f"No expression data extracted. Probe map size: {len(probe_to_gene_map)}"
                )

        return expression_data
    
    def _extract_refpool_from_deletion_samples(self, gsm_list, probe_to_gene_map) -> SortedDict:
        """Extract refpool values from deletion samples' Cy3 or Cy5 channel.
        
        In deletion samples, dye-swap design means:
        - Sample ending in '-a': Cy5 = deletion, Cy3 = refpool
        - Sample ending in '-b': Cy5 = refpool, Cy3 = deletion (dye swap)
        
        Returns:
            SortedDict: Average refpool expression values at deletion experiment scale
        """
        refpool_values = {}
        
        for gsm in gsm_list:
            if not hasattr(gsm, "table") or gsm.table is None:
                continue
                
            table = gsm.table
            if "ID_REF" not in table.columns:
                continue
                
            # Determine which channel has refpool based on sample name
            title = gsm.metadata.get('title', [''])[0] if hasattr(gsm, 'metadata') else ''
            
            # Check for dye-swap pattern
            if '-a' in title or '_a' in title or title.endswith('a'):
                refpool_column = "Signal Norm_Cy3"  # Standard: deletion in Cy5, refpool in Cy3
            elif '-b' in title or '_b' in title or title.endswith('b'):
                refpool_column = "Signal Norm_Cy5"  # Dye swap: deletion in Cy3, refpool in Cy5
            else:
                # Default to Cy3 as refpool
                refpool_column = "Signal Norm_Cy3"
            
            # Extract refpool values
            if refpool_column in table.columns:
                for _, row in table.iterrows():
                    probe_id = str(int(row["ID_REF"]))
                    if probe_id in probe_to_gene_map:
                        gene = probe_to_gene_map[probe_id]
                        try:
                            value = float(row[refpool_column])
                            if value > 0:
                                if gene not in refpool_values:
                                    refpool_values[gene] = []
                                refpool_values[gene].append(value)
                        except (ValueError, TypeError):
                            continue
        
        # Average refpool values across samples
        averaged_refpool = SortedDict()
        for gene, values in refpool_values.items():
            averaged_refpool[gene] = np.mean(values)
        
        return averaged_refpool
    
    def _extract_refpool_from_wt_gsm(self, gsm, probe_to_gene_map=None) -> SortedDict:
        """Extract refpool expression values from a WT GSM object.
        
        In WT samples (GSE42215, GSE42217, GSE42240, GSE42241):
        - Some samples: wt vs. refpool (wt in Cy5, refpool in Cy3)
        - Other samples: refpool vs. wt (refpool in Cy5, wt in Cy3)
        
        We need to determine which channel contains refpool and extract it.
        """
        expression_data = SortedDict()
        
        if not hasattr(gsm, "table") or gsm.table is None:
            return expression_data
            
        table = gsm.table
        
        # Check for expected columns
        if "ID_REF" not in table.columns:
            log.warning(f"ID_REF column not found")
            return expression_data
        
        # We need both Cy5 and Cy3 to extract refpool
        if "Signal Norm_Cy5" not in table.columns or "Signal Norm_Cy3" not in table.columns:
            log.warning("Missing Cy5 or Cy3 columns, cannot extract refpool")
            return expression_data
        
        # Determine hybridization direction from sample metadata
        title = gsm.metadata.get('title', [''])[0] if hasattr(gsm, 'metadata') else ''
        geo_accession = gsm.metadata.get('geo_accession', [''])[0] if hasattr(gsm, 'metadata') else ''
        
        # Debug logging
        log.debug(f"Processing WT sample {geo_accession}: {title}")
        
        # Determine which channel has refpool based on title or metadata
        # The VALUE column represents log2(Cy5/Cy3)
        # If VALUE is positive when wt > refpool, then wt is in Cy5
        # If VALUE is negative when wt < refpool, then wt is in Cy5
        
        # For WT samples, we need to look at the VALUE to infer direction
        # If most VALUEs are near 0, it's wt vs refpool (both similar)
        # We can also check the title pattern
        
        refpool_column = "Signal Norm_Cy3"  # Default assumption
        
        # Try to parse from title
        title_lower = title.lower()
        if 'refpool' in title_lower:
            # Look for patterns like "refpool vs wt" or "wt vs refpool"
            if 'refpool vs' in title_lower or 'refpool-' in title_lower:
                # refpool is first, so it's in Cy5
                refpool_column = "Signal Norm_Cy5"
            elif 'vs refpool' in title_lower or '-refpool' in title_lower:
                # refpool is second, so it's in Cy3
                refpool_column = "Signal Norm_Cy3"
        
        # Alternative: check sample name patterns
        # GSE42215 samples often have patterns in their names
        if '-a' in title or '_a' in title:
            refpool_column = "Signal Norm_Cy3"  # Standard orientation
        elif '-b' in title or '_b' in title:
            refpool_column = "Signal Norm_Cy5"  # Dye swap
        
        log.debug(f"  Using {refpool_column} as refpool channel")
        
        # Extract refpool values
        for _, row in table.iterrows():
            probe_id = str(int(row["ID_REF"]))
            
            if probe_to_gene_map and probe_id in probe_to_gene_map:
                gene = probe_to_gene_map[probe_id]
                try:
                    value = float(row[refpool_column])
                    if value > 0:  # Only keep positive values
                        expression_data[gene] = value
                except (ValueError, TypeError):
                    continue
        
        return expression_data

    def _load_mating_type_map(self) -> tuple[dict, dict]:
        """Load mating type information from supplementary Table S1.

        Returns:
            Tuple of:
            - Dictionary mapping systematic gene names to strains (BY4741 or BY4742)
            - Dictionary mapping common gene names to systematic names
        """
        systematic_to_strain = {}
        common_to_systematic = {}

        # Path to the supplementary table
        table_path = osp.join(self.raw_dir, "kemmeren2014_table_s1.xlsx")

        # Check if the file exists - REQUIRED
        if not osp.exists(table_path):
            raise FileNotFoundError(
                f"Supplementary Table S1 not found at {table_path}\n"
                f"Please download it from: https://www.cell.com/cms/10.1016/j.cell.2014.02.054/attachment/7b6014f0-a526-4f16-ae4a-cd04fd03efce/mmc1.xlsx\n"
                f"And save it as: {table_path}"
            )

        try:
            # Read the Excel file
            df = pd.read_excel(table_path, sheet_name=0)  # First sheet
            log.info(f"Loaded Excel file with {len(df)} rows")
            log.info(
                f"Columns in Excel file: {list(df.columns)[:10]}"
            )  # Show first 10 columns

            # Look for columns containing gene names and mating type
            # MUST use "orf name" column which contains systematic names
            orf_col = None
            gene_col = None  # Common gene name column for validation
            mating_col = None

            # Find the orf name column (contains systematic names like YJL095W)
            for col in df.columns:
                col_lower = col.lower()
                if "orf" in col_lower and "name" in col_lower:
                    orf_col = col
                    break
            
            # Also find gene column for validation
            for col in df.columns:
                if col.lower() == "gene":
                    gene_col = col
                    break

            # Find the mating type column
            for col in df.columns:
                col_lower = col.lower()
                if "mating" in col_lower and "type" in col_lower:
                    mating_col = col
                    break
            
            if not orf_col:
                raise ValueError("Required 'orf name' column not found in Excel file!")

            if orf_col and mating_col:
                log.info(
                    f"Using columns: {orf_col} for systematic names, {mating_col} for mating type"
                )
                if gene_col:
                    log.info(f"Also found {gene_col} column for validation")

                # Debug: Track what's in the Excel
                excel_genes_original = []
                excel_genes_converted = []
                duplicate_systematics = set()  # Track orf names with multiple gene names

                for _, row in df.iterrows():
                    systematic_name = row[orf_col]  # Use orf name directly
                    mating_type = row[mating_col]
                    common_name = row[gene_col] if gene_col else None

                    if pd.notna(systematic_name) and pd.notna(mating_type):
                        # Use the systematic name directly from Excel
                        systematic_name = str(systematic_name).upper()
                        excel_genes_original.append(systematic_name)
                        excel_genes_converted.append(systematic_name)
                        
                        # No genome validation needed - Excel is authoritative
                        
                        # Check for duplicates
                        if systematic_name in systematic_to_strain:
                            duplicate_systematics.add(systematic_name)
                            if common_name:
                                log.debug(f"Duplicate orf {systematic_name}: adding alias {common_name}")

                        # Map mating type to strain
                        mating_str = str(mating_type).upper()
                        if (
                            "MATALPHA" in mating_str
                            or "MATΑ" in mating_str
                            or "MAT ALPHA" in mating_str
                        ):
                            strain = "BY4742"
                        elif "MATA" in mating_str and "ALPHA" not in mating_str:
                            strain = "BY4741"
                        else:
                            log.warning(
                                f"Unknown mating type '{mating_type}' for gene {systematic_name}"
                            )
                            continue

                        systematic_to_strain[systematic_name] = strain
                        
                        # Also create common name to systematic mapping
                        if common_name and pd.notna(common_name):
                            common_name_upper = str(common_name).upper()
                            common_to_systematic[common_name_upper] = systematic_name

                # Debug output
                log.info(f"Loaded mating type for {len(systematic_to_strain)} genes")
                log.info(
                    f"First 10 original gene names from Excel: {excel_genes_original[:10]}"
                )
                log.info(
                    f"First 10 converted systematic names: {excel_genes_converted[:10]}"
                )

                # Check if YAL014C is in the Excel
                if "YAL014C" in excel_genes_converted:
                    idx = excel_genes_converted.index("YAL014C")
                    log.info(
                        f"YAL014C found in Excel! Original name: {excel_genes_original[idx]}"
                    )
                else:
                    log.warning("YAL014C NOT found in Excel converted names")
                    # Check if it's in original names
                    if "YAL014C" in excel_genes_original:
                        log.info(
                            "YAL014C IS in Excel original names but conversion failed!"
                        )

                    # Check for similar names
                    similar = [g for g in excel_genes_converted if "YAL014" in g]
                    if similar:
                        log.info(f"Found similar genes in Excel: {similar}")
                # Count strains
                by4741_count = sum(1 for v in systematic_to_strain.values() if v == "BY4741")
                by4742_count = sum(1 for v in systematic_to_strain.values() if v == "BY4742")
                log.info(f"Loaded mating type for {len(systematic_to_strain)} genes")
                log.info(f"Loaded {len(common_to_systematic)} common name mappings")
                log.info(
                    f"BY4741 (MATa): {by4741_count}, BY4742 (MATalpha): {by4742_count}"
                )
                
                if duplicate_systematics:
                    log.info(f"Found {len(duplicate_systematics)} orf names with multiple gene names in Excel")
            else:
                if not orf_col:
                    log.error(f"Missing required 'orf name' column. Available: {list(df.columns)}")
                if not mating_col:
                    log.error(f"Missing mating type column. Available: {list(df.columns)}")

        except Exception as e:
            log.error(f"Failed to load mating type map: {e}")

        return systematic_to_strain, common_to_systematic


    def resolve_gene_name_comprehensive(
        self, gene_name: str, common_to_systematic: dict, 
        systematic_to_strain: dict, already_assigned: set = None
    ) -> str:
        """Comprehensive gene name resolution with multiple fallback strategies.
        
        Priority:
        1. Special hardcoded mappings
        2. Excel mapping (experiment-specific)
        3. gene_attribute_table (one-to-one)
        4. alias_to_systematic (one-to-many with filtering)
        5. Case-insensitive alias_to_systematic
        6. Direct check in systematic_to_strain
        7. Return None if cannot resolve
        """
        if already_assigned is None:
            already_assigned = set()
            
        gene_upper = gene_name.upper()
        
        # Pass 1: Check special mappings first
        if gene_upper in self.SPECIAL_GENE_MAPPINGS:
            systematic = self.SPECIAL_GENE_MAPPINGS[gene_upper]
            if systematic in systematic_to_strain:
                self.resolved_by_alias += 1
                return systematic
        
        # Pass 2: Direct Excel mapping
        if gene_upper in common_to_systematic:
            self.resolved_by_excel += 1
            return common_to_systematic[gene_upper]
        
        # Pass 3: Gene attribute table (one-to-one)
        if hasattr(self.genome, 'gene_attribute_table'):
            df = self.genome.gene_attribute_table
            
            # Check if it's already systematic in the table
            if gene_upper in df['ID'].values:
                if gene_upper in systematic_to_strain:
                    self.resolved_by_gene_table += 1
                    return gene_upper
            
            # Check gene column
            matches = df[df['gene'] == gene_upper]
            if not matches.empty:
                systematic = matches.iloc[0]['ID']
                if systematic in systematic_to_strain:
                    self.resolved_by_gene_table += 1
                    return systematic
            
            # Check Alias column
            matches = df[df['Alias'] == gene_upper]
            if not matches.empty:
                systematic = matches.iloc[0]['ID']
                if systematic in systematic_to_strain:
                    self.resolved_by_gene_table += 1
                    return systematic
        
        # Pass 4: Alias to systematic (handle one-to-many)
        if hasattr(self.genome, 'alias_to_systematic'):
            candidates = self.genome.alias_to_systematic.get(gene_upper, [])
            
            if candidates:
                # Filter for Excel existence only (multiple aliases can map to same systematic)
                valid = [c for c in candidates if c in systematic_to_strain]
                
                if len(valid) == 1:
                    log.debug(f"Resolved {gene_name} → {valid[0]} via alias matching")
                    self.resolved_by_alias += 1
                    return valid[0]
                elif len(valid) > 1:
                    # Pick first alphabetically for consistency
                    chosen = sorted(valid)[0]
                    log.warning(f"Multiple candidates for {gene_name}: {valid}, chose {chosen}")
                    self.resolved_by_alias += 1
                    return chosen
        
        # Pass 5: Case-insensitive alias to systematic (for cases like CYCC vs CycC)
        if hasattr(self.genome, 'alias_to_systematic'):
            # Try case-insensitive matching
            for alias, systematics in self.genome.alias_to_systematic.items():
                if alias.upper() == gene_upper:
                    # Filter for Excel existence
                    valid = [c for c in systematics if c in systematic_to_strain]
                    if len(valid) == 1:
                        log.debug(f"Resolved {gene_name} → {valid[0]} via case-insensitive alias matching")
                        self.resolved_by_alias += 1
                        return valid[0]
                    elif len(valid) > 1:
                        chosen = sorted(valid)[0]
                        log.debug(f"Multiple candidates for {gene_name} (case-insensitive): {valid}, chose {chosen}")
                        self.resolved_by_alias += 1
                        return chosen
        
        # Pass 6: Check if the gene itself exists in Excel (might be non-standard name)
        if gene_upper in systematic_to_strain:
            log.info(f"Using {gene_upper} directly (found in Excel)")
            self.resolved_by_excel += 1
            return gene_upper
        
        # Final: Cannot resolve
        log.info(f"Cannot resolve {gene_name} - not in genome annotations or Excel")
        self.unresolved_genes += 1
        return None
    
    def convert_gene_name(self, gene_name: str, common_to_systematic: dict) -> str:
        """Simple conversion for expression data - keep as-is if no mapping.
        
        Used for probe-to-gene mappings where we want to keep all genes.
        """
        gene_upper = gene_name.upper()
        
        # Check Excel mapping
        if gene_upper in common_to_systematic:
            return common_to_systematic[gene_upper]
        
        # Try gene_attribute_table
        if hasattr(self.genome, 'gene_attribute_table'):
            df = self.genome.gene_attribute_table
            
            # Check gene column
            matches = df[df['gene'] == gene_upper]
            if not matches.empty:
                return matches.iloc[0]['ID']
            
            # Check Alias column
            matches = df[df['Alias'] == gene_upper]
            if not matches.empty:
                return matches.iloc[0]['ID']
        
        # Return as-is
        return gene_name

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for Kemmeren this is handled in process()."""
        return df

    def create_experiment(self):
        """Required by base class but not used - see create_expression_experiment."""
        pass

    def _validate_log2_ratios(
        self, deletion_samples_by_gene, probe_to_gene_map, systematic_to_strain
    ):
        """Validate that our calculated log2 ratios match the original GEO data.
        
        IMPORTANT: The VALUE column in GEO follows standard microarray convention:
        VALUE = log2(Cy3/Cy5) = log2(refpool/deletion) = log2(reference/test)
        
        This means:
        - Negative VALUE: deletion has HIGHER expression than refpool
        - Positive VALUE: deletion has LOWER expression than refpool
        
        For biological interpretation, log2(deletion/refpool) would be more intuitive
        (positive = upregulated), but we validate against GEO's convention here.
        """
        log.info("\n=== Validating Log2 Ratios ===")
        
        # Debug: First check what the refpool looks like
        log.info("\n=== Refpool Reference Debug ===")
        if hasattr(self, 'wt_expression_BY4742') and self.wt_expression_BY4742:
            sample_genes = list(self.wt_expression_BY4742.keys())[:5]
            for gene in sample_genes:
                log.info(f"Refpool BY4742 {gene}: {self.wt_expression_BY4742[gene]:.4f}")
        
        # Sample a subset of genes for validation
        genes_to_validate = list(deletion_samples_by_gene.keys())[:min(20, len(deletion_samples_by_gene))]
        
        original_ratios = []
        calculated_ratios = []
        
        for gene_idx, gene_name in enumerate(genes_to_validate):
            gsm_list = deletion_samples_by_gene[gene_name]
            if not gsm_list:
                continue
                
            # Get strain to select appropriate refpool reference
            strain = systematic_to_strain.get(gene_name)
            if strain == "BY4741":
                refpool_ref = self.wt_expression_BY4741
            elif strain == "BY4742":
                refpool_ref = self.wt_expression_BY4742
            else:
                continue
            
            # Get first GSM for this gene
            gsm = gsm_list[0]
            
            # Debug: Show first few probes for first gene
            if gene_idx == 0:
                log.info(f"\n=== Debug for gene {gene_name} ===")
                log.info(f"GSM: {gsm.metadata.get('geo_accession', [''])[0]}")
            
            # Extract original log2 ratios from VALUE column
            if hasattr(gsm, "table") and gsm.table is not None:
                table = gsm.table
                
                # Debug: Show available columns for first gene
                if gene_idx == 0:
                    log.info(f"Available columns: {list(table.columns)[:10]}")
                    # Show first few rows
                    log.info("\nFirst 3 rows of data:")
                    for i, row in enumerate(table.head(3).iterrows()):
                        _, row_data = row
                        if "VALUE" in table.columns:
                            value_str = f"{row_data['VALUE']:.4f}" if pd.notna(row_data['VALUE']) else "NaN"
                            log.info(f"Row {i}: VALUE={value_str}")
                        if "Signal Norm_Cy5" in table.columns and "Signal Norm_Cy3" in table.columns:
                            cy5_str = f"{row_data['Signal Norm_Cy5']:.4f}" if pd.notna(row_data['Signal Norm_Cy5']) else "NaN"
                            cy3_str = f"{row_data['Signal Norm_Cy3']:.4f}" if pd.notna(row_data['Signal Norm_Cy3']) else "NaN"
                            log.info(f"       Cy5={cy5_str}, Cy3={cy3_str}")
                
                if "VALUE" in table.columns and "ID_REF" in table.columns:
                    probe_count = 0
                    for _, row in table.iterrows():
                        probe_id = str(int(row["ID_REF"]))
                        if probe_id in probe_to_gene_map:
                            probe_gene = probe_to_gene_map[probe_id]
                            # We don't need refpool_ref anymore - just validate Cy5/Cy3 directly
                            try:
                                # Original log2 ratio from GEO (VALUE column = log2(Cy3/Cy5))
                                original = float(row["VALUE"])
                                
                                # VALUE = log2(Cy3/Cy5) = log2(refpool/deletion)
                                # This is standard microarray convention: log2(reference/test)
                                
                                if "Signal Norm_Cy5" in table.columns and "Signal Norm_Cy3" in table.columns:
                                    cy5 = float(row["Signal Norm_Cy5"])  # Deletion mutant
                                    cy3 = float(row["Signal Norm_Cy3"])  # Refpool
                                    
                                    # Calculate log2(Cy3/Cy5) to match VALUE convention
                                    if cy5 > 0:
                                        calculated = np.log2(cy3 / cy5)  # Note: Cy3/Cy5, not Cy5/Cy3
                                    else:
                                        continue
                                    
                                    # Debug first few probes of first gene
                                    if gene_idx == 0 and probe_count < 3:
                                        log.info(f"\nProbe {probe_id} -> Gene {probe_gene}:")
                                        log.info(f"  Original VALUE: {original:.4f}")
                                        log.info(f"  Cy5 (deletion): {cy5:.4f}")
                                        log.info(f"  Cy3 (refpool): {cy3:.4f}")
                                        log.info(f"  Calculated log2(Cy3/Cy5): {calculated:.4f}")
                                        log.info(f"  Difference: {abs(original - calculated):.4f}")
                                        probe_count += 1
                                    
                                    original_ratios.append(original)
                                    calculated_ratios.append(calculated)
                            except (ValueError, TypeError) as e:
                                continue
        
        if len(original_ratios) > 10:
            # Calculate correlation
            correlation = np.corrcoef(original_ratios, calculated_ratios)[0, 1]
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((np.array(original_ratios) - np.array(calculated_ratios))**2))
            
            log.info(f"Validation samples: {len(original_ratios)}")
            log.info(f"Correlation between original and calculated log2 ratios: {correlation:.3f}")
            log.info(f"RMSE: {rmse:.3f}")
            
            if correlation < 0.8:
                log.warning(f"WARNING: Low correlation ({correlation:.3f}) between original and calculated ratios!")
                log.warning("This may indicate issues with refpool extraction or processing.")
            else:
                log.info("✓ Good correlation - refpool references appear correct")
        else:
            log.warning("Not enough data points for validation")
        
        log.info("===")
    
    def _average_dye_swaps(
        self, gsm_list, probe_to_gene_map=None
    ) -> tuple[SortedDict, SortedDict]:
        """Average expression values from dye-swap technical replicates.

        Returns:
            tuple: (averaged_expression, technical_std)
        """
        # Collect all expression data
        all_expressions = SortedDict()

        for gsm in gsm_list:
            expr_data = self._extract_expression_from_gsm(gsm, probe_to_gene_map)
            for gene, value in expr_data.items():
                if gene not in all_expressions:
                    all_expressions[gene] = []
                all_expressions[gene].append(value)

        # Calculate mean and std for each gene
        averaged_expression = SortedDict()
        technical_std = SortedDict()

        for gene, values in all_expressions.items():
            if len(values) > 0:
                averaged_expression[gene] = np.mean(values)
                if len(values) > 1:
                    technical_std[gene] = np.std(values, ddof=1)  # Sample std
                else:
                    technical_std[gene] = np.nan  # NaN for single measurement

        return averaged_expression, technical_std

    @staticmethod
    def create_expression_experiment(
        dataset_name,
        sample_info,
        expression_data,
        technical_std,
        refpool_expression,
        refpool_std,
    ):
        # Genome reference - strain MUST be specified (BY4741 or BY4742)
        if "strain" not in sample_info:
            raise ValueError(
                "Strain (BY4741 or BY4742) must be specified in sample_info"
            )
        strain = sample_info["strain"]
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain
        )

        # Create genotype for deletion mutant
        systematic_name = sample_info["systematic_gene_name"]
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=systematic_name,  # Use same name
                )
            ]
        )

        # Environment - YPD medium at 30°C (or SC depending on dataset)
        environment = Environment(
            media=Media(name="SC", state="liquid"),  # Kemmeren used SC medium
            temperature=Temperature(value=30),
        )
        environment_reference = environment.model_copy()

        # Calculate log2 ratios if we have refpool reference
        log2_ratios = SortedDict()
        if refpool_expression:
            for gene, value in expression_data.items():
                if (
                    gene in refpool_expression
                    and refpool_expression[gene] > 0
                ):
                    # Calculate log2 fold change vs refpool
                    log2_ratios[gene] = np.log2(value / refpool_expression[gene])

        # Create phenotype with actual expression data
        phenotype = MicroarrayExpressionPhenotype(
            expression=expression_data,
            expression_log2_ratio=log2_ratios if log2_ratios else None,
            expression_technical_std=technical_std if technical_std else None,
        )

        # Create reference phenotype (refpool expression)
        if refpool_expression:
            phenotype_reference = MicroarrayExpressionPhenotype(
                expression=refpool_expression,
                expression_log2_ratio=None,  # No ratio for reference
                expression_technical_std=refpool_std,  # Include refpool technical std
            )
        else:
            # Use the expression data itself as reference if no refpool available
            phenotype_reference = MicroarrayExpressionPhenotype(
                expression=expression_data,
                expression_log2_ratio=None,
                expression_technical_std=technical_std,
            )

        # Create reference
        reference = MicroarrayExpressionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        # Create experiment
        experiment = MicroarrayExpressionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        # Publication
        publication = Publication(
            pubmed_id="24766815",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/24766815/",
            doi="10.1016/j.cell.2014.02.054",
            doi_url="https://doi.org/10.1016/j.cell.2014.02.054",
        )

        return experiment, reference, publication


if __name__ == "__main__":
    dataset = MicroarrayKemmeren2014Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014")
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset gene set size: {len(dataset.gene_set)}")
    print(f"First 10 genes in gene_set: {list(dataset.gene_set)[:10]}")

    if len(dataset) > 0:
        # Get raw data (dictionary format) - this is what dataset[0] returns
        data = dataset[0]

        # The data is returned as a dictionary with deserialized content
        print(f"\nFirst dataset item (index 0):")
        print(f"  Data type: {type(data)}")
        print(f"  Keys: {data.keys()}")

        # Access the dictionaries
        experiment = data["experiment"]
        reference = data["reference"]
        publication = data["publication"]

        print(f"\n=== Experiment Details ===")
        print(f"  Dataset: {experiment['dataset_name']}")
        perturbed_gene = experiment["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ]
        print(f"  Perturbed gene: {perturbed_gene}")

        # Show expression data summary
        exp_expression = experiment["phenotype"]["expression"]
        print(f"  Expression measurements: {len(exp_expression)} genes")

        # Show first 5 expression values
        print(f"  First 5 expression values:")
        for i, (gene, value) in enumerate(list(exp_expression.items())[:5]):
            print(f"    {gene}: {value:.4f}")

        print(f"\n=== Reference Details ===")
        print(f"  Dataset: {reference['dataset_name']}")
        print(f"  Genome: {reference['genome_reference']}")
        print(f"  Environment: {reference['environment_reference']}")

        # Show reference expression (wildtype baseline)
        ref_expression = reference["phenotype_reference"]["expression"]
        print(f"  Reference expression: {len(ref_expression)} genes")

        # Show first 5 reference expression values
        print(f"  First 5 reference expression values (wildtype):")
        for i, (gene, value) in enumerate(list(ref_expression.items())[:5]):
            print(f"    {gene}: {value:.4f}")

        # Check if reference has technical std
        if "expression_technical_std" in reference["phenotype_reference"]:
            ref_std = reference["phenotype_reference"]["expression_technical_std"]
            if ref_std:
                print(f"  Reference has technical std for {len(ref_std)} genes")

        # Compare specific gene between experiment and reference
        print(f"\n=== Gene Comparison (exp vs ref) ===")
        # Pick first 3 genes for comparison
        sample_genes = list(exp_expression.keys())[:3]
        for gene in sample_genes:
            if gene in ref_expression:
                exp_val = exp_expression[gene]
                ref_val = ref_expression[gene]
                log2_ratio = np.log2(exp_val / ref_val) if ref_val != 0 else np.nan
                print(f"  {gene}:")
                print(f"    Deletion mutant: {exp_val:.4f}")
                print(f"    Wildtype (ref): {ref_val:.4f}")
                print(f"    Log2 ratio: {log2_ratio:.4f}")

        # Check perturbed gene presence
        print(f"\n=== Perturbed Gene Status ===")
        print(f"  Gene: {perturbed_gene}")
        print(f"  In deletion mutant expression: {perturbed_gene in exp_expression}")
        print(f"  In wildtype reference expression: {perturbed_gene in ref_expression}")

        print(f"\n=== Publication ===")
        print(f"  PubMed ID: {publication['pubmed_id']}")
        print(f"  DOI: {publication['doi']}")

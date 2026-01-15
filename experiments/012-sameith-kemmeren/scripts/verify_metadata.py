# experiments/012-sameith-kemmeren/scripts/verify_metadata
# [[experiments.012-sameith-kemmeren.scripts.verify_metadata]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/verify_metadata
# Test file: experiments/012-sameith-kemmeren/scripts/test_verify_metadata.py

"""
Metadata Verification Script for Kemmeren & Sameith Datasets

Verifies metadata consistency across three microarray datasets:
- Kemmeren2014: ~1,484 single deletion mutants
- SmMicroarraySameith2015: 82 single deletion mutants (GSTFs)
- DmMicroarraySameith2015: ~72 double deletion mutants (GSTF pairs)

Expected outcomes:
- All reference log2 ratios = 0.0 ± 1e-6
- All environments: SC liquid media at 30°C
- Correct strain distributions
- No NaN in expression values
"""

import os
import os.path as osp
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict

# Removed timestamp import - using stable filenames instead
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.datasets.scerevisiae.sameith2015 import (
    SmMicroarraySameith2015Dataset,
    DmMicroarraySameith2015Dataset,
)

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

# Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "10"))
LOG2_TOLERANCE = 1e-6  # Tolerance for floating-point comparison

# Setup logging (no timestamp - stable filenames for documentation)
log_file = osp.join(
    EXPERIMENT_ROOT, "012-sameith-kemmeren/results", "verification.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_data_structure(data, dataset_name, idx):
    """Validate expected data structure before processing."""
    required_keys = ["experiment", "reference", "publication"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"{dataset_name}[{idx}] missing key: {key}")

    # Check experiment structure
    experiment = data["experiment"]
    if "phenotype" not in experiment:
        raise ValueError(f"{dataset_name}[{idx}] missing phenotype in experiment")
    if "genotype" not in experiment:
        raise ValueError(f"{dataset_name}[{idx}] missing genotype in experiment")

    # Check reference structure
    reference = data["reference"]
    required_ref_keys = [
        "phenotype_reference",
        "environment_reference",
        "genome_reference",
    ]
    for key in required_ref_keys:
        if key not in reference:
            raise ValueError(f"{dataset_name}[{idx}] missing {key} in reference")

    # Check phenotype structure
    phenotype = experiment["phenotype"]
    if "expression_log2_ratio" not in phenotype:
        raise ValueError(f"{dataset_name}[{idx}] missing expression_log2_ratio")

    return True


def verify_reference_log2_ratios(data, dataset_name, idx, anomalies):
    """Verify that reference log2 ratios are all 0.0 within tolerance."""
    ref_log2_ratios = data["reference"]["phenotype_reference"]["expression_log2_ratio"]

    violations = []
    for gene, ratio in ref_log2_ratios.items():
        if abs(ratio) > LOG2_TOLERANCE:
            violations.append((gene, ratio))

    if violations:
        anomalies["reference_log2_violations"].append(
            {
                "dataset": dataset_name,
                "index": idx,
                "violation_count": len(violations),
                "sample_violations": violations[:5],  # First 5 examples
            }
        )
        return False, len(violations)

    return True, 0


def verify_environment(data, dataset_name, idx, anomalies):
    """Verify environment is SC liquid media at 30°C."""
    env_ref = data["reference"]["environment_reference"]

    media_name = env_ref.get("media", {}).get("name", "")
    media_state = env_ref.get("media", {}).get("state", "")
    temperature = env_ref.get("temperature", {}).get("value", None)

    issues = []

    if media_name != "SC":
        issues.append(f"media_name={media_name} (expected 'SC')")
    if media_state != "liquid":
        issues.append(f"media_state={media_state} (expected 'liquid')")
    if temperature != 30.0:
        issues.append(f"temperature={temperature} (expected 30.0)")

    if issues:
        anomalies["environment_violations"].append(
            {"dataset": dataset_name, "index": idx, "issues": issues}
        )
        return False

    return True


def verify_expression_completeness(data, dataset_name, idx, anomalies):
    """Verify no NaN values in expression data."""
    expression = data["experiment"]["phenotype"]["expression"]

    nan_count = sum(1 for v in expression.values() if np.isnan(v))
    total_genes = len(expression)

    if nan_count > 0:
        nan_percentage = (nan_count / total_genes) * 100
        anomalies["expression_nan"].append(
            {
                "dataset": dataset_name,
                "index": idx,
                "nan_count": nan_count,
                "total_genes": total_genes,
                "nan_percentage": nan_percentage,
            }
        )
        return False, total_genes, nan_count

    return True, total_genes, 0


def get_strain(data):
    """Extract strain from reference genome."""
    return data["reference"]["genome_reference"].get("strain", "UNKNOWN")


def process_dataset(dataset, dataset_name, sample_range=None):
    """Process a single dataset and collect statistics."""
    logger.info(f"Processing {dataset_name} with {len(dataset)} experiments")

    # Initialize statistics
    stats = {
        "dataset_name": dataset_name,
        "total_experiments": 0,
        "strain_by4741_count": 0,
        "strain_by4742_count": 0,
        "strain_unknown_count": 0,
        "reference_log2_all_zeros": True,
        "reference_log2_violations": 0,
        "media_consistent": True,
        "temperature_consistent": True,
        "expression_complete": True,
        "expression_gene_counts": [],
        "expression_nan_counts": [],
        "total_anomalies": 0,
    }

    # Anomaly tracking
    anomalies = {
        "reference_log2_violations": [],
        "environment_violations": [],
        "expression_nan": [],
    }

    # Determine iteration range
    if sample_range is None:
        iter_range = range(len(dataset))
    else:
        iter_range = range(min(sample_range, len(dataset)))

    # Process each experiment
    for i in tqdm(iter_range, desc=f"Verifying {dataset_name}"):
        try:
            data = dataset[i]

            # Validate structure
            validate_data_structure(data, dataset_name, i)

            # Count experiments
            stats["total_experiments"] += 1

            # Verify strain
            strain = get_strain(data)
            if strain == "BY4741":
                stats["strain_by4741_count"] += 1
            elif strain == "BY4742":
                stats["strain_by4742_count"] += 1
            else:
                stats["strain_unknown_count"] += 1
                logger.warning(f"{dataset_name}[{i}]: Unknown strain '{strain}'")

            # Verify reference log2 ratios
            log2_ok, violation_count = verify_reference_log2_ratios(
                data, dataset_name, i, anomalies
            )
            if not log2_ok:
                stats["reference_log2_all_zeros"] = False
                stats["reference_log2_violations"] += violation_count

            # Verify environment
            env_ok = verify_environment(data, dataset_name, i, anomalies)
            if not env_ok:
                stats["media_consistent"] = False
                stats["temperature_consistent"] = False

            # Verify expression completeness
            expr_ok, gene_count, nan_count = verify_expression_completeness(
                data, dataset_name, i, anomalies
            )
            stats["expression_gene_counts"].append(gene_count)
            if not expr_ok:
                stats["expression_complete"] = False
                stats["expression_nan_counts"].append(nan_count)

        except Exception as e:
            logger.error(f"Error processing {dataset_name}[{i}]: {e}")
            stats["total_anomalies"] += 1

    # Calculate summary statistics
    stats["expression_gene_count_mean"] = np.mean(stats["expression_gene_counts"])
    stats["expression_gene_count_std"] = np.std(stats["expression_gene_counts"], ddof=1)
    stats["total_anomalies"] = (
        len(anomalies["reference_log2_violations"])
        + len(anomalies["environment_violations"])
        + len(anomalies["expression_nan"])
    )

    return stats, anomalies


def write_anomaly_report(all_anomalies, output_path):
    """Write detailed anomaly report to text file."""
    from datetime import datetime
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("METADATA VERIFICATION ANOMALY REPORT\n")
        f.write(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for dataset_name, anomalies in all_anomalies.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'=' * 80}\n\n")

            # Reference log2 violations
            if anomalies["reference_log2_violations"]:
                f.write(f"\n--- Reference Log2 Ratio Violations ---\n")
                f.write(
                    f"Total experiments with violations: {len(anomalies['reference_log2_violations'])}\n\n"
                )
                for violation in anomalies["reference_log2_violations"][:10]:
                    f.write(f"  Index {violation['index']}:\n")
                    f.write(f"    Violation count: {violation['violation_count']}\n")
                    f.write(f"    Sample violations:\n")
                    for gene, ratio in violation["sample_violations"]:
                        f.write(f"      {gene}: {ratio}\n")
                    f.write("\n")

            # Environment violations
            if anomalies["environment_violations"]:
                f.write(f"\n--- Environment Violations ---\n")
                f.write(
                    f"Total experiments with violations: {len(anomalies['environment_violations'])}\n\n"
                )
                for violation in anomalies["environment_violations"][:10]:
                    f.write(f"  Index {violation['index']}:\n")
                    for issue in violation["issues"]:
                        f.write(f"    - {issue}\n")
                    f.write("\n")

            # Expression NaN issues
            if anomalies["expression_nan"]:
                f.write(f"\n--- Expression NaN Issues ---\n")
                f.write(
                    f"Total experiments with NaN: {len(anomalies['expression_nan'])}\n\n"
                )
                for issue in anomalies["expression_nan"][:10]:
                    f.write(f"  Index {issue['index']}:\n")
                    f.write(f"    NaN count: {issue['nan_count']}\n")
                    f.write(f"    Total genes: {issue['total_genes']}\n")
                    f.write(f"    NaN percentage: {issue['nan_percentage']:.2f}%\n")
                    f.write("\n")

            if not any(anomalies.values()):
                f.write("\n✓ No anomalies detected for this dataset!\n")


def main():
    logger.info("=" * 80)
    logger.info("METADATA VERIFICATION SCRIPT")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    if DEBUG_MODE:
        logger.info(f"Sample size: {SAMPLE_SIZE}")
    logger.info("=" * 80)

    # Load datasets
    logger.info("\n--- Loading Datasets ---")

    kemmeren = MicroarrayKemmeren2014Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
        io_workers=0,
    )
    logger.info(f"Loaded Kemmeren2014: {len(kemmeren)} experiments")

    sm_sameith = SmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        io_workers=0,
    )
    logger.info(f"Loaded SmSameith2015: {len(sm_sameith)} experiments")

    dm_sameith = DmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015"),
        io_workers=0,
    )
    logger.info(f"Loaded DmSameith2015: {len(dm_sameith)} experiments")

    # Process datasets
    logger.info("\n--- Processing Datasets ---")
    sample_range = SAMPLE_SIZE if DEBUG_MODE else None

    kemmeren_stats, kemmeren_anomalies = process_dataset(
        kemmeren, "Kemmeren2014", sample_range
    )
    sm_stats, sm_anomalies = process_dataset(sm_sameith, "SmSameith2015", sample_range)
    dm_stats, dm_anomalies = process_dataset(dm_sameith, "DmSameith2015", sample_range)

    # Create summary DataFrame
    summary_df = pd.DataFrame([kemmeren_stats, sm_stats, dm_stats])

    # Reorder columns for readability
    column_order = [
        "dataset_name",
        "total_experiments",
        "strain_by4741_count",
        "strain_by4742_count",
        "strain_unknown_count",
        "reference_log2_all_zeros",
        "reference_log2_violations",
        "media_consistent",
        "temperature_consistent",
        "expression_complete",
        "expression_gene_count_mean",
        "expression_gene_count_std",
        "total_anomalies",
    ]
    summary_df = summary_df[column_order]

    # Save outputs (no timestamp - stable filenames for documentation)
    output_dir = osp.join(EXPERIMENT_ROOT, "012-sameith-kemmeren/results")
    os.makedirs(output_dir, exist_ok=True)

    summary_path = osp.join(output_dir, "metadata_verification_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n✓ Summary saved to: {summary_path}")

    anomalies_path = osp.join(output_dir, "metadata_verification_anomalies.txt")
    all_anomalies = {
        "Kemmeren2014": kemmeren_anomalies,
        "SmSameith2015": sm_anomalies,
        "DmSameith2015": dm_anomalies,
    }
    write_anomaly_report(all_anomalies, anomalies_path)
    logger.info(f"✓ Anomalies report saved to: {anomalies_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    print("\n" + summary_df.to_string(index=False))
    logger.info("\n" + "=" * 80)

    # Print overall status
    all_passed = (
        summary_df["reference_log2_all_zeros"].all()
        and summary_df["media_consistent"].all()
        and summary_df["temperature_consistent"].all()
        and summary_df["expression_complete"].all()
        and (summary_df["strain_unknown_count"] == 0).all()
    )

    if all_passed:
        logger.info("✓ ALL CHECKS PASSED!")
    else:
        logger.warning("⚠ SOME CHECKS FAILED - Review anomalies report")

    logger.info(f"\nLog file: {log_file}")


if __name__ == "__main__":
    main()

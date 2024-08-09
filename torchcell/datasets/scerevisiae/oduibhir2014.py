import GEOparse
import pandas as pd


def main():
    # Load the GEO dataset
    gse = GEOparse.get_GEO(filepath="/Users/michaelvolk/Downloads/GSE42527_family.soft")

    # Access the sample data and pivot to get the VALUE matrix
    samples_df = gse.pivot_samples("VALUE")

    # Access the probe information
    probe_info_df = gse.gpls["GPL11232"].table

    # Merge the sample data with the probe information based on probe ID
    merged_df = pd.merge(
        samples_df,
        probe_info_df[["PROBE_ID", "SYSTEMATIC_NAME"]],
        left_index=True,
        right_on="PROBE_ID",
    )

    # Pivot the table so that samples are rows, genes are columns, and values are the expression levels
    sxn_table = merged_df.pivot_table(
        index="SAMPLE_ID", columns="SYSTEMATIC_NAME", values="VALUE"
    )

    # Create a dictionary of sample IDs and their titles
    sample_titles = {gsm: gse.gsms[gsm].metadata["title"][0] for gsm in sxn_table.index}

    # Add the sample titles as the first column in the table
    sxn_table.insert(0, "Sample_Title", sxn_table.index.map(sample_titles))

    # Print or return the resulting DataFrame
    print(sxn_table.head())


if __name__ == "__main__":
    main()

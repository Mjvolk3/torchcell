import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadr
import numpy as np


# Function to calculate the percentage of genes outside a certain range around 1
def calc_percentage(df, lower_bound, upper_bound):
    count = 0
    total = df.size
    for col in df.columns:
        count += ((df[col] > upper_bound) | (df[col] <= lower_bound)).sum()
    return (count / total) * 100


# Read RDS file into a DataFrame
def read_rds_with_pyreadr():
    rds_file_path = "../Gene_Graph/data/Mechanistic_Aware/expressionOnly.RDS"
    result = pyreadr.read_r(rds_file_path)
    df = result[None]
    return df


if __name__ == "__main__":
    df = read_rds_with_pyreadr()
    df = df.drop(columns=["log2relT"])

    # Compute percentages
    perc_25 = calc_percentage(df, 0.75, 1.25)
    perc_50 = calc_percentage(df, 0.5, 1.5)
    perc_75 = calc_percentage(df, 0.25, 1.75)
    perc_100 = calc_percentage(df, 0, 2)

    # Melt the DataFrame to long format
    df_melted = df.melt(var_name="Gene", value_name="Expression")

    # Create the box plot without outliers
    plt.figure(figsize=(100, 20))
    ax = sns.boxplot(
        x="Gene",
        y="Expression",
        data=df_melted,
        showfliers=False,
        whiskerprops={"color": "#E84A26"},
    )

    # Remove the caps
    for cap in ax.lines:
        if cap.get_linestyle() == "None":
            cap.set_visible(False)

    # Change the thickness of the axis border (spines)
    for spine in ax.spines.values():
        spine.set_linewidth(10)  # Sets the thickness of the spines

    # Customizations
    ax.set(xticklabels=[])
    plt.xlabel("SMF genes", fontsize=80)
    plt.ylabel("Expression Fold Change", fontsize=80)
    plt.yticks(fontsize=80)
    ax.tick_params(axis="y", length=20, width=10)

    # Title
    plt.title(
        f"SMF Gene Expression Box Plots ({df.shape[1]} Genes, {df.shape[0]} Single Mutants)",
        fontsize=100,
    )

    # Add text annotations for percentages
    text_str = f"±0.25 fold change = ~{int(perc_25*df.shape[1]/100)} genes ({perc_25:.2f}%) \n±0.50 fold change = ~{int(perc_50*df.shape[1]/100)} genes ({perc_50:.2f}%) \n±0.75 fold change = ~{int(perc_75*df.shape[1]/100)} genes ({perc_75:.2f}%) \n±1.00 fold change = ~{int(perc_100*df.shape[1]/100)} genes ({perc_100:.2f}%)"
    ax.text(
        0.5,
        0.9,
        text_str,
        transform=ax.transAxes,
        fontsize=80,
        verticalalignment="top",
        horizontalalignment="center",
    )

    # Save the figure
    plt.savefig(
        f"SMF_gene_expression_box_plot_{df.shape[1]}_genes_{df.shape[0]}_sm.png"
    )

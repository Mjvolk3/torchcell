import csv
import json
import os.path as osp

# Saccharomyces cerevisiae Raw Data Extraction Functions
from typing import Optional

import networkx as nx
import pandas as pd
import torch
from goatools.obo_parser import GODag
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.utils.convert import from_networkx

import torchcell.yeastmine as ym

# TODO write a data loader that can download the below files into ../data/
# This works already... just need to fully implement.
# download data from data_url and put in "data/Baryshnikova2010"
# import requests
# import os.path as osp

# url = "https://research.cs.umn.edu/csbio/SGAScore/Supplementary_data_1_SMF_standard_100209.xls"
# resp = requests.get(url)
# base_path = "data/Baryshnikova2010"
# file_name = url.split("/")[-1]
# with open(osp.join(base_path, file_name), "wb") as f:
#     f.write(resp.content)


def read_raw_data(
    singles: bool = True, digenic: bool = True, trigenic: bool = True
) -> dict:
    """[read raw data from data directory]

    Args:
        singles (bool, optional): [Include data]. Defaults to True.
        digenic (bool, optional): [Include data]. Defaults to True.
        trigenic (bool, optional): [Include Data]. Defaults to True.

    Returns:
        dict: [Keys are names of dataframes. Items are corresponding dataframe of raw fitness data.]
    """
    # initialize to None
    df_singles = None
    df_digenic_damp = None
    df_digenic_exe = None
    df_digenic_exn_nxe = None
    df_digenic_nxn = None
    df_trigenic = None

    df_names = []
    dfs = []
    # Read in Raw Data
    if singles:
        df_singles = pd.read_excel(
            r"data/Baryshnikova2010/10.1038-nmeth.1534_S1_Single-mutant_fitness_standard.xls",
            header=0,
            names=["gene", "fitness_mean", "fitness_std"],
        )
        df_names.append("df_singles")
        dfs.append(df_singles)
    if digenic:
        df_digenic_damp = pd.read_csv(
            r"data/TheCellMap/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_DAmP.txt",
            sep="\t",
        )
        df_names.append("df_digenic_damp")
        dfs.append(df_digenic_damp)
        df_digenic_exe = pd.read_csv(
            r"data/TheCellMap/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_ExE.txt",
            sep="\t",
        )
        df_names.append("df_digenic_exe")
        dfs.append(df_digenic_exe)
        df_digenic_exn_nxe = pd.read_csv(
            r"data/TheCellMap/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_ExN_NxE.txt",
            sep="\t",
        )
        df_names.append("df_digenic_exn_nxe")
        dfs.append(df_digenic_exn_nxe)
        df_digenic_nxn = pd.read_csv(
            r"data/TheCellMap/Data File S1. Raw genetic interaction datasets: Pair-wise interaction format/SGA_NxN.txt",
            sep="\t",
        )
        df_names.append("df_digenic_nxn")
        dfs.append(df_digenic_nxn)
        dfs.append(
            pd.concat(
                [df_digenic_damp, df_digenic_exe, df_digenic_exn_nxe, df_digenic_nxn]
            ).reset_index()
        )
        df_names.append("df_digenic_all")
    if trigenic:
        df_trigenic = pd.read_csv(
            r"data/Kuzmin2018/aao1729_Data_S1.tsv", sep="\t", encoding="utf-8"
        )
        df_names.append("df_trigenic")
        dfs.append(df_trigenic)

    raw_data = dict(
        zip(
            df_names,
            dfs,
        )
    )
    return raw_data


def get_gene_list(
    singles: bool = False, digenic: bool = True, trigenic: bool = True
) -> list:
    """[Get list of all gene names contained in dataframes. Intended to be able to get list of genes from all or subset of dataframes]

    Args:
        raw_data (dict): [Keys are names of dataframes. Items are data frames.]

    Returns:
        list: [Union of genes names from different dataframes]
    """
    gene_list_file = "data/preprocessed/gene_list.csv"
    if osp.exists(gene_list_file):
        with open("data/preprocessed/gene_list.csv", "r") as f:
            g_list = csv.reader(f, delimiter=",")
            gene_list = [row for row in g_list][0]
    else:
        raw_data = read_raw_data(singles, digenic, trigenic)
        data_name_list = []
        # Singles Data
        if "df_singles" in raw_data.keys():
            singles_names = (
                raw_data["df_singles"]["gene"].str.split("_", expand=True)[0].to_list()
            )
            data_name_list.append(pd.Series(singles_names))
        # Digenic Data
        if "df_digenic_damp" in raw_data.keys():
            digenic_damp_names = _ext_digenic_genes(raw_data["df_digenic_damp"])
            data_name_list.append(digenic_damp_names)
        if "df_digenic_exe" in raw_data.keys():
            digenic_exe_names = _ext_digenic_genes(raw_data["df_digenic_exe"])
            data_name_list.append(digenic_exe_names)
        if "df_digenic_exn_nxe" in raw_data.keys():
            digenc_exn_nxe_names = _ext_digenic_genes(raw_data["df_digenic_exn_nxe"])
            data_name_list.append(digenc_exn_nxe_names)
        if "df_digenic_nxn" in raw_data.keys():
            digenic_nxn_names = _ext_digenic_genes(raw_data["df_digenic_nxn"])
            data_name_list.append(digenic_nxn_names)
        # Trigenic Data
        if "df_trigenic" in raw_data.keys():
            trigenic_names = _ext_trigenic_genes(raw_data["df_trigenic"])
            data_name_list.append(trigenic_names)
        # Concatenate all data
        total_name_instances = pd.concat(data_name_list)
        gene_list = sorted(list(total_name_instances.drop_duplicates()))
        # first two genes not in SGD database, last three genes have stop codons in sequence, all have no, or few GO terms
        ignored_genes = get_ignored_genes()
        gene_list = [gene for gene in gene_list if gene not in ignored_genes]
        with open(gene_list_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(gene_list)
    return gene_list


def get_gene_id_translation_table():
    file_path = "data/preprocessed/gene_id_translation_table.csv"
    if osp.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        print("NEED TO IMPLEMENT METHOD")
        df = None
    return df


def get_ignored_genes():
    ignored_genes = ["YIL080W", "YAR037W", "YAR040C", "YFL057C", "YFL056C"]
    return ignored_genes


def get_smf_bary(write_json=True, to_tensor=False) -> list:
    # TODO docstring
    data_path = "data/preprocessed/gene_singles_fitness_bary.json"
    if osp.exists(data_path):
        with open(data_path, "r") as f:
            gene_singles = json.load(f)
        if to_tensor:
            gene_singles = [val[0] for _, val in gene_singles]
            gene_singles = torch.tensor(list(gene_singles), dtype=torch.float)
        return gene_singles
    else:
        df = read_raw_data(True, False, False)["df_singles"]
        # only take full knockout alleles
        reg_allele_filter = [
            i == 1 for i in [len(i) for i in df["gene"].str.split("_")]
        ]
        df_reg_allele = df[reg_allele_filter]
        names = df_reg_allele["gene"].str.split("_", expand=True)[0].to_list()
        labels = df_reg_allele["fitness_mean"].to_list()
        label_data = [list(i) for i in zip(names, labels)]
        label_data = {i[0]: i[1] for i in label_data}
        # read gene_essentiality_raw.json to list of lists -> raw null mutant
        # TODO call lethality.whatever
        with open("data/preprocessed/gene_essentiality_raw.json", "r") as f:
            essential_genes = json.load(f)
        gene_singles = get_gene_list()
        gene_singles = {i: None for i in gene_singles}
        gene_list = get_gene_list()
        problem_label = {}
        for gene in gene_list:
            if essential_genes[gene] == "viable":
                try:
                    gene_singles[gene] = label_data[gene]
                except KeyError:
                    print("Warning! viable without fitness label")
                    problem_label[gene] = "viable"
                    # Cannot assign any value
                    gene_singles[gene] = -1
            elif essential_genes[gene] == "inviable":
                try:
                    gene_singles[gene] = label_data[gene]
                    problem_label[gene] = "inviable with fitness label"
                except KeyError:
                    gene_singles[gene] = 0
            elif essential_genes[gene] == "unknown":
                try:
                    gene_singles[gene] = label_data[gene]
                except KeyError:
                    print("Warning! unknown without fitness label")
                    problem_label[gene] = "unknown without fitness label"
                    # Cannot assign any value
                    gene_singles[gene] = -1
        # handling inviables
        problem_inviables = {
            k: v for k, v in problem_label.items() if v == "inviable with fitness label"
        }
        inviables_fitness = {
            gene: label_data[gene] for gene in problem_inviables.keys()
        }
        print(f"Fitness of null mutant for SGD 'inviable':\n {inviables_fitness}")
        gene_singles = [[[k], [v]] for k, v in gene_singles.items()]
        if write_json:
            # write gene_singles to json file in data/preprocessed
            with open(data_path, "w") as f:
                json.dump(gene_singles, f, indent=4)
        if to_tensor:
            gene_singles = [val[0] for _, val in gene_singles]
            gene_singles = torch.tensor(list(gene_singles), dtype=torch.float)
        return gene_singles


def get_double_fitness():
    pass


def get_triple_fitness():
    pass


# Hidden Functions
def _ext_digenic_genes(df_digenic: pd.DataFrame) -> pd.Series:
    """[Extract all digenic gene name instances from digenic interaction data]

    Args:
        df_digenic (pd.DataFrame): [digenic interaction data from data/TheCellMap]

    Returns:
        pd.Series: [(nx1) gene names with duplicates. Duplicates represent number of mutant instances]
    """
    qry = df_digenic["Query Strain ID"].str.split("_", expand=True)[0]
    arr = df_digenic["Array Strain ID"].str.split("_", expand=True)[0]
    names = pd.concat([qry, arr])
    return names


def _ext_trigenic_genes(df_trigenic: pd.DataFrame) -> pd.Series:
    """[Extract all trigenic gene name instances from trigenic interaction data]

    Args:
        df_trigenic (pd.DataFrame): [trigenic interaction data from data/Kuzmin2018]

    Returns:
        pd.Series: [(nx1) gene names with duplicates. Duplicates represent number of mutant instances]
    """
    qry_split = (
        df_trigenic["Query strain ID"]
        .str.split("_", expand=True)[0]
        .str.split("+", expand=True)
    )
    qry = pd.concat([qry_split[0], qry_split[1]])
    arr = df_trigenic["Array strain ID"].str.split("_", expand=True)[0]
    names = pd.concat([qry, arr])
    return names


if __name__ == "__main__":
    pass

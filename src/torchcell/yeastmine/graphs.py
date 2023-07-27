# %%
import json
import os
import os.path as osp

import networkx as nx
import numpy as np
import pandas as pd
import torch
from gene_graph.sc_graph import get_gene_list
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm


# TODO check if theses are actually DiGraphs.... when they were read in I thought they were multigraphs... check for all.
#
##############################--Encoders--####################################
class CategoricalEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_feature_dims):
        super(CategoricalEncoder, self).__init__()
        self.embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.embedding_list[i](x[:, i])

        return x_embedding


##############################--Encoders--####################################
##############################--Edge_dicts--####################################
def regulators_edge_dict() -> nx.DiGraph:
    print("Writing regulators features to graph")
    gene_list = get_gene_list()
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    for gene in tqdm(gene_list):
        with open(f"data/preprocessed/node_ym_attrs/{gene}.json", "r") as f:
            data = json.load(f)
            regulators = [attrs["regulators"] for attrs in data.values()][0]
            # initialize lists
            regulators_identifier = []
            pubmed_id = []
            strain_background = []
            regulation_direction = []
            regulator_type = []
            annotation_type = []
            for reg in regulators:
                for k, v in reg.items():
                    if k == "regulatoryRegions.regulator.secondaryIdentifier":
                        regulators_identifier.append(v)
                    elif k == "regulatoryRegions.publications.pubMedId":
                        pubmed_id.append(v)
                    elif k == "regulatoryRegions.strainBackground":
                        strain_background.append(v)
                    elif k == "regulatoryRegions.annotationType":
                        annotation_type.append(v)
                    elif k == "regulatoryRegions.regulatorType":
                        regulator_type.append(v)
                    elif k == "regulatoryRegions.regulationDirection":
                        regulation_direction.append(v)

            for reg in zip(
                regulators_identifier,
                pubmed_id,
                strain_background,
                annotation_type,
                regulator_type,
                regulation_direction,
            ):
                if gene in G.nodes and reg[0] in G.nodes:
                    # Default order arranged according to tie_break_popular function
                    edge_dict = {
                        "pubmed_id": reg[1],
                        "strain_background": reg[2],
                        "annotation_type": reg[3],
                        "regulator_type": reg[4],
                        "regulation_direction": reg[5],
                    }
                    # removing None and replacing with string "None"
                    edge_dict = {k: v if v else "None" for k, v in edge_dict.items()}
                    # direction is regulator -> queried gene
                    G.add_edge(reg[0], gene, **edge_dict)
    return G


def protein_interactions_edge_dict() -> nx.DiGraph:
    print("Writing protein interactions features to graph")
    gene_list = get_gene_list()
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    for gene in tqdm(gene_list):
        with open(f"data/preprocessed/node_ym_attrs/{gene}.json", "r") as f:
            data = json.load(f)
            physical_interactions = [
                attrs["physical_interactions"] for attrs in data.values()
            ][0]
            # initialize lists
            participant = []
            experiment_name = []
            annotation_type = []
            role = []
            detection_method_identifier = []
            for ppi in physical_interactions:
                for k, v in ppi.items():
                    if k == "interactions.participant2.secondaryIdentifier":
                        participant.append(v)
                    elif k == "interactions.details.experiment.name":
                        experiment_name.append(v)
                    elif k == "interactions.details.annotationType":
                        annotation_type.append(v)
                    elif (
                        k
                        == "interactions.details.experiment.interactionDetectionMethods.identifier"
                    ):
                        detection_method_identifier.append(v)
                    elif k == "interactions.details.role1":
                        role.append(v)
            for ppi in zip(
                participant,
                experiment_name,
                annotation_type,
                detection_method_identifier,
                role,
            ):
                if gene in G.nodes and ppi[0] in G.nodes:
                    # Default order arranged according to tie_break_popular function
                    edge_dict = {
                        "experiment_name": ppi[1],
                        "annotation_type": ppi[2],
                        "detection_method_identifier": ppi[3],
                        "role": ppi[4],
                    }
                    # Replace None with 'None'
                    edge_dict = {k: v if v else "None" for k, v in edge_dict.items()}
                    # Can Add ordering option.
                    # example
                    # desired_order_list = [5, 2, 4, 3, 1]
                    # reordered_dict = {k: sample_dict[k] for k in desired_order_list}
                    # direction is gene -> participant
                    G.add_edge(gene, ppi[0], **edge_dict)
    return G


def gene_interactions_edge_dict() -> nx.DiGraph:
    print("Writing gene interactions features to graph")
    gene_list = get_gene_list()
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    for gene in tqdm(gene_list):
        with open(f"data/preprocessed/node_ym_attrs/{gene}.json", "r") as f:
            data = json.load(f)
            gene_interactions = [attrs["gene_interactions"] for attrs in data.values()][
                0
            ]
            participant = []
            experiment_name = []
            interaction_details_annotation_type = []
            interactions_details_phenotype = []
            interactions_details_role1 = []
            interactions_detection_methods_identifier = []
            p_value = []
            sgaScore = []
            for ggi in gene_interactions:
                for k, v in ggi.items():
                    if k == "interactions.participant2.secondaryIdentifier":
                        participant.append(v)
                    elif k == "interactions.details.experiment.name":
                        experiment_name.append(v)
                    elif k == "interactions.details.annotationType":
                        interaction_details_annotation_type.append(v)
                    elif (
                        k
                        == "interactions.details.experiment.interactionDetectionMethods.identifier"
                    ):
                        interactions_detection_methods_identifier.append(v)
                    elif k == "interactions.details.phenotype":
                        interactions_details_phenotype.append(v)
                    elif k == "interactions.details.role1":
                        interactions_details_role1.append(v)
                    elif k == "interactions.alleleinteractions.pvalue":
                        p_value.append(v)
                    elif k == "interactions.alleleinteractions.sgaScore":
                        sgaScore.append(v)

            for ggi in zip(
                participant,
                experiment_name,
                interaction_details_annotation_type,
                interactions_detection_methods_identifier,
                interactions_details_phenotype,
                interactions_details_role1,
                p_value,
                sgaScore,
            ):
                if gene in G.nodes and ggi[0] in G.nodes:
                    edge_dict = {
                        "experiment_name": ggi[1],
                        "interaction_details_annotation_type": ggi[2],
                        "interactions_detection_methods_identifier": ggi[3],
                        "interactions_details_phenotype": ggi[4],
                        "interactions_details_role1": ggi[5],
                        "p_value": ggi[6],
                        "sgaScore": ggi[7],
                    }
                    # handling None values
                    # replace None values in edge dict with 'None'
                    edge_dict = {k: v if v else "None" for k, v in edge_dict.items()}
                    # direction is gene -> participant... This should not matter for the gene graph because it should be symmetric
                    G.add_edge(gene, ggi[0], **edge_dict)
    return G


##############################--Edge_dicts--####################################
##############################--Miscellaneous--#################################
# This comes from n53... this entire file of functions still needs to be added
def agg_edge_data(G: nx.Graph) -> pd.DataFrame:
    # loop over G and print target, source, key, and data
    sources = []
    targets = []
    keys = []
    datas = []
    for source, target, key, data in G.edges(data=True, keys=True):
        sources.append(source)
        targets.append(target)
        keys.append(key)
        datas.append(data)

    df = pd.DataFrame(
        {"source": sources, "target": targets, "keys": keys, "data": datas}
    )
    df = (
        df.groupby(["source", "target"])["data"]
        .apply(list)
        .reset_index(name="edge_data")
    )
    df["edge_data_size"] = df["edge_data"].transform(len)
    return df


# TODO This belongs in other label data file
def get_synthetic_lethal() -> list:
    print("Getting synthetic lethality from yeastmined features")
    gene_list = get_gene_list()
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    synthetic_lethal = []
    for gene in tqdm(gene_list):
        with open(f"data/preprocessed/node_ym_attrs/{gene}.json", "r") as f:
            data = json.load(f)
            gene_interactions = [attrs["gene_interactions"] for attrs in data.values()][
                0
            ]
            participant = []
            interactions_details_phenotype = []
            for ggi in gene_interactions:
                for k, v in ggi.items():
                    if k == "interactions.participant2.secondaryIdentifier":
                        participant.append(v)
                    elif k == "interactions.details.phenotype":
                        if v == "inviable":
                            interactions_details_phenotype.append(v)

            for ggi in zip(
                participant,
                interactions_details_phenotype,
            ):
                if gene in G.nodes and ggi[0] in G.nodes:
                    edge_dict = {
                        "interactions_details_phenotype": ggi[1],
                    }
                    genes_inviable = [
                        [[gene, ggi[0]], [v]] for k, v in edge_dict.items()
                    ]
                    synthetic_lethal.extend(genes_inviable)
    names = [i[0] for i in synthetic_lethal]
    df = pd.DataFrame(names)
    df.loc[df[0] > df[1], "names_sort"] = df[0] + "_" + df[1]
    df.loc[df[0] < df[1], "names_sort"] = df[1] + "_" + df[0]
    gene_0 = df.loc[df["names_sort"].drop_duplicates().index][0].to_list()
    gene_1 = df.loc[df["names_sort"].drop_duplicates().index][1].to_list()
    synthetic_lethal_clean = []
    for gene in zip(gene_0, gene_1):
        synthetic_lethal_clean.append([[gene[0], gene[1]], ["inviable"]])
    synthetic_lethal_clean
    return synthetic_lethal_clean


##############################--Miscellaneous--#################################
##############################--Node_attrs--####################################
def protein_half_life_df(gene_list: list) -> tuple:
    print("protein half life features:")
    df_protein_half_life = pd.DataFrame()
    for gene in tqdm(gene_list):
        with open(f"data/preprocessed/node_ym_attrs/{gene}.json", "r") as f:
            data = json.load(f)
        protein_half_life = [attrs["protein_half_life"] for attrs in data.values()][0]
        df = pd.DataFrame(protein_half_life)
        df_protein_half_life = pd.concat([df_protein_half_life, df])
    df_protein_half_life = df_protein_half_life[
        [
            "secondaryIdentifier",
            "proteins.proteinHalfLife.value",
            "proteins.proteinHalfLife.units",
            "proteins.proteinHalfLife.publication.pubMedId",
        ]
    ]
    df_clean = df_protein_half_life.copy()
    df_clean.loc[
        df_clean["proteins.proteinHalfLife.units"] == "min",
        "proteins.proteinHalfLife.value",
    ] = (
        df_protein_half_life[
            df_protein_half_life["proteins.proteinHalfLife.units"] == "min"
        ]["proteins.proteinHalfLife.value"]
        / 60
    )
    df_clean.loc[
        df_clean["proteins.proteinHalfLife.units"] == "min",
        "proteins.proteinHalfLife.units",
    ] = "hr"
    # can drop duplicates since rows are identical
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean[["secondaryIdentifier", "proteins.proteinHalfLife.value"]]
    df_fill_small = pd.DataFrame(
        df_clean[["proteins.proteinHalfLife.value"]].median()
    ).rename(columns={0: "fill_value"})
    fill_genes = list(set(gene_list) - set(df_clean["secondaryIdentifier"].to_list()))
    df_fill = pd.DataFrame(
        {
            "secondaryIdentifier": fill_genes,
            "proteins.proteinHalfLife.value": df_fill_small.loc[
                "proteins.proteinHalfLife.value"
            ].to_list()
            * len(fill_genes),
        }
    )
    return df_clean, df_fill


def median_protein_abundance_df(gene_list: list) -> tuple:
    print("median protein abundance features:")
    df_median_protein_abundance = pd.DataFrame()
    for gene in tqdm(gene_list):
        with open(f"data/preprocessed/node_ym_attrs/{gene}.json", "r") as f:
            data = json.load(f)
        median_protein_abundance = [
            attrs["median_protein_abundance"] for attrs in data.values()
        ][0]
        df = pd.DataFrame(median_protein_abundance)
        df_median_protein_abundance = pd.concat([df_median_protein_abundance, df])
    df_median_protein_abundance = df_median_protein_abundance.loc[
        (~df_median_protein_abundance["proteins.median"].isna())
    ]
    df_clean = df_median_protein_abundance.copy()
    df_clean.loc[
        df_clean["proteins.MAD"].isna(), "proteins.MAD"
    ] = df_median_protein_abundance["proteins.MAD"].mean()
    df_clean["proteins.MAD"].isna().sum()
    df_clean = df_clean[
        ["secondaryIdentifier", "qualifier", "proteins.median", "proteins.MAD"]
    ].drop_duplicates()
    qualifier = (
        pd.DataFrame(df_clean["qualifier"].value_counts().sort_values(ascending=False))
        .reset_index()
        .loc[0, "index"]
    )
    fill = df_clean[["proteins.median", "proteins.MAD"]].median()
    fill["qualifier"] = qualifier
    df_fill_small = pd.DataFrame({"fill_value": fill})
    fill_genes = list(set(gene_list) - set(df_clean["secondaryIdentifier"].to_list()))
    df_fill = pd.DataFrame(
        {
            "secondaryIdentifier": fill_genes,
            "proteins.median": df_fill_small.loc["proteins.median"].to_list()
            * len(fill_genes),
            "proteins.MAD": df_fill_small.loc["proteins.MAD"].to_list()
            * len(fill_genes),
            "qualifier": df_fill_small.loc["qualifier"].to_list() * len(fill_genes),
        }
    )
    return df_clean, df_fill


def chromosomal_location_df(gene_list: list) -> tuple:
    print("chromosomal location features:")
    df_chromosomal_location = pd.DataFrame()
    for gene in tqdm(gene_list):
        with open(f"data/preprocessed/node_ym_attrs/{gene}.json", "r") as f:
            data = json.load(f)
        chromosomal_location = [
            attrs["chromosomal_location"] for attrs in data.values()
        ][0]
        df = pd.DataFrame(chromosomal_location)
        df_chromosomal_location = pd.concat([df_chromosomal_location, df])
    df_chromosomal_location[
        df_chromosomal_location["secondaryIdentifier"].duplicated(keep=False)
    ].sort_values("secondaryIdentifier")
    df_clean = df_chromosomal_location.drop_duplicates()[
        [
            "secondaryIdentifier",
            "chromosome.primaryIdentifier",
            "chromosomeLocation.start",
            "chromosomeLocation.end",
            "chromosomeLocation.strand",
        ]
    ]
    df_gene_list = pd.DataFrame({"secondaryIdentifier": gene_list})
    df_clean = pd.merge(df_clean, df_gene_list, on="secondaryIdentifier", how="inner")
    return df_clean


def gene_node_dict(write_graph: bool = True) -> nx.DiGraph:
    file_name = "data/preprocessed/gene_reprs/yeastmine/node_reprs.gpickle"
    if osp.exists(file_name):
        print(f"reading graph from: {file_name}")
        G = nx.read_gpickle(file_name)
    else:
        print("Writing gene node features to graph")
        gene_list = get_gene_list()
        G = nx.Graph()
        G.add_nodes_from(gene_list)
        # features
        df_clean, df_fill = median_protein_abundance_df(gene_list)
        df_median_protein_abundance = pd.concat([df_clean, df_fill])
        df_clean, df_fill = protein_half_life_df(gene_list)
        df_protein_half_life = pd.concat([df_clean, df_fill])
        df_chromosomal_location = chromosomal_location_df(gene_list)
        # merge features
        df_temp = pd.merge(
            df_median_protein_abundance,
            df_protein_half_life,
            on="secondaryIdentifier",
            how="outer",
        )
        df_temp = pd.merge(
            df_temp, df_chromosomal_location, on="secondaryIdentifier", how="outer"
        )
        # write data to graph
        for row in df_temp.iterrows():
            gene = row[1]["secondaryIdentifier"]
            data = (row[1].drop("secondaryIdentifier")).to_dict()
            G.add_node(gene, **data)
        if write_graph:
            nx.write_gpickle(G, file_name)
    return G


###
# TODO add n53--graph_statistics.py
# - mainly need to look for writing "_digraph.gpickle"
###


def add_name_rename(G: nx.Graph):
    attrs = {}
    mapping = {}
    for i, name in enumerate(list(G.nodes())):
        attrs[name] = {"gene_name": name}
        mapping[name] = i
    nx.set_node_attributes(G, attrs)
    H = nx.relabel_nodes(G, mapping)
    return H


# TODO include ordinal encoder and lookup encoder
def node_reprs_to_data(
    enc: str = "lookup",
    emb_dim: int = 2,
    attr_included_names: list = None,
    save: bool = True,
):
    # One hot encoding protein interactions
    # default behavior is to include the all one_hot features inside conditional
    # names of features that could be potentially used
    file_name_torch = "data/preprocessed/gene_reprs/yeastmine/node_reprs.pt"
    # if osp.exists(file_name_torch):
    #     Gt = torch.load(file_name_torch)
    # else:
    attr_data_names = [
        "qualifier",
        "proteins.median",
        "proteins.MAD",
        "proteins.proteinHalfLife.value",
        "chromosome.primaryIdentifier",
        "chromosomeLocation.start",
        "chromosomeLocation.end",
        "chromosomeLocation.strand",
    ]

    if attr_included_names is None:
        attr_included_names = attr_data_names
    file_name_nx = "data/preprocessed/gene_reprs/yeastmine/node_reprs.gpickle"
    G = nx.read_gpickle(file_name_nx)
    G = add_name_rename(G)
    Gt = from_networkx(G)
    data = []
    for name in attr_included_names:
        if type(Gt[name]) != list:
            data.append(Gt[name].numpy())
        else:
            data.append(Gt[name])
    data = np.array(data).T
    df = pd.DataFrame(data)
    # "None" string used previous for plotting... this graph shouldn't have any nan values
    df.replace("None", np.nan, inplace=True)
    # replacements have already been done prior to this function.
    df.columns = attr_data_names
    df = df.astype(
        {
            "proteins.median": "float",
            "proteins.MAD": "float",
            "proteins.proteinHalfLife.value": "float",
            "chromosomeLocation.start": "float",
            "chromosomeLocation.end": "float",
        }
    )
    numeric_names = [
        "proteins.median",
        "proteins.MAD",
        "proteins.proteinHalfLife.value",
        "chromosomeLocation.start",
        "chromosomeLocation.end",
    ]
    # treat numeric features
    numeric_names_input = list(set(numeric_names).intersection(set(attr_data_names)))
    attr_numeric = df[numeric_names_input].to_numpy()
    attr_numeric = torch.tensor(attr_numeric)
    # treat one hot features
    categorical_names = [
        "qualifier",
        "chromosome.primaryIdentifier",
        "chromosomeLocation.strand",
    ]
    categorical_names_input = list(
        set(categorical_names).intersection(set(attr_data_names))
    )
    attr_one_hot = df[categorical_names_input].to_numpy()
    ###

    if (enc == "ordinal") or (enc == "lookup"):
        edge_attr_categorical = df[categorical_names_input].to_numpy()
        enc_ordinal = OrdinalEncoder()
        enc_ordinal.fit(edge_attr_categorical)
        edge_attr_categorical = enc_ordinal.transform(edge_attr_categorical)
        edge_attr_categorical_tensor = torch.tensor(edge_attr_categorical)
        if enc == "lookup":
            full_feature_dims = []
            for i in df[categorical_names_input]:
                # Get feature dims list
                # reference https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
                full_feature_dims.append(df[i].value_counts().shape[0])
            enc_lookup = CategoricalEncoder(
                emb_dim=emb_dim, full_feature_dims=full_feature_dims
            )
            edge_attr_categorical_tensor = enc_lookup(
                edge_attr_categorical_tensor.long()
            )
    elif enc == "onehot":
        # "None" string used previous for plotting
        df.replace("None", np.nan, inplace=True)
        edge_attr_categorical = (
            df[categorical_names_input].replace(np.nan, "None").to_numpy()
        )
        # Not sparse to so can be converted to torch tensor
        enc = OneHotEncoder(sparse=False)
        enc.fit(edge_attr_categorical)
        edge_attr_categorical_tensor = torch.tensor(
            enc.transform(edge_attr_categorical)
        )
    ###

    # combine one_hot and numeric
    attr_tensor = torch.concat((edge_attr_categorical_tensor, attr_numeric), dim=1)
    # concatenate tensors
    Gt.x = attr_tensor
    edge_attr_drop_names = attr_data_names
    # remove named features and keep one_hot features
    for name in edge_attr_drop_names:
        Gt[name] = None
    if save:
        torch.save(Gt, file_name_torch)
    return Gt


def unirep_to_data():
    gene_list = get_gene_list()
    base_path = "data/preprocessed/gene_reprs/unirep"
    file_names = os.listdir(base_path)
    assert len(file_names) == len(gene_list), "gene list and unirep genes do not match!"
    data = Data()
    dim = len(torch.load(osp.join(base_path, file_names[0])))
    x = torch.empty(len(file_names), dim)
    gene_name = []
    for i, gene in enumerate(gene_list):
        gene_name.append(gene)
        x[i] = torch.load(f"data/preprocessed/gene_reprs/unirep/{gene}.pt")
    data.x = x
    data.edge_index = torch.empty(2, 0)
    data.gene_name = gene_name
    data.num_nodes = len(gene_name)
    return data


def join_edge_node_reprs(data_edge: Data, data_node: Data):
    H = Data()
    if data_edge.gene_name == data_node.gene_name:
        print("joining data")
        H.x = data_node.x
        H.edge_index = data_edge.edge_index
        H.edge_attr = data_edge.edge_attr
        H.gene_name = data_edge.gene_name
        H.num_nodes = data_edge.num_nodes
    else:
        print("data did not have matching gene_name")
        H = None
    return H


##############################--Node_attrs--####################################
##############################--Edge_attrs--######################################


def nx_regulators_to_torch(
    enc: str = "lookup",
    emb_dim: int = 2,
    attr_included_names: list = None,
    save=True,
):
    # enc: ["lookup", "ordinal","one_hot"]
    # One hot encoding regulators
    # default behavior is to include the all one_hot features inside conditional
    edge_attr_data_names = [
        "strain_background",
        "annotation_type",
        "regulator_type",
        "regulation_direction",
    ]
    if attr_included_names is None:
        attr_included_names = edge_attr_data_names
    file_name = "data/preprocessed/graphs/regulators/raw/regulators_digraph.gpickle"
    G = nx.read_gpickle(file_name)
    G = add_name_rename(G)
    Gt = from_networkx(G)
    data = []
    for name in attr_included_names:
        data.append(Gt[name])
    data = np.array(data).T
    df = pd.DataFrame(data)
    if (enc == "ordinal") or (enc == "lookup"):
        edge_attr = df.to_numpy()
        enc_ordinal = OrdinalEncoder()
        enc_ordinal.fit(edge_attr)
        edge_attr = enc_ordinal.transform(edge_attr)
        edge_attr_tensor = torch.tensor(edge_attr)
        if enc == "lookup":
            full_feature_dims = []
            # Note on all ggi and nodes features need [categorical_names_input]
            for i in df:
                # Get feature dims list
                # reference https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
                full_feature_dims.append(df[i].value_counts().shape[0])
            enc_lookup = CategoricalEncoder(
                emb_dim=emb_dim, full_feature_dims=full_feature_dims
            )
            edge_attr_tensor = enc_lookup(edge_attr_tensor.long())
    elif enc == "onehot":
        # "None" string used previous for plotting
        df.replace("None", np.nan, inplace=True)
        edge_attr = df.to_numpy()
        # Not sparse to so can be converted to torch tensor
        enc = OneHotEncoder(sparse=False)
        enc.fit(edge_attr)
        edge_attr_tensor = torch.tensor(enc.transform(edge_attr))
    Gt.edge_attr = edge_attr_tensor
    edge_attr_drop_names = [
        "strain_background",
        "annotation_type",
        "regulator_type",
        "regulation_direction",
    ] + ["pubmed_id"]
    # remove named features and keep one_hot features
    for name in edge_attr_drop_names:
        Gt[name] = None
    if save:
        torch.save(Gt, file_name.split(".")[0] + ".pt")
    return Gt


def nx_protein_interactions_to_torch(
    enc: str = "lookup",
    emb_dim: int = 2,
    attr_included_names: list = None,
    save=True,
):
    # One hot encoding protein interactions
    # default behavior is to include the all one_hot features inside conditional
    # names of features that could be potentially used
    edge_attr_data_names = ["annotation_type", "detection_method_identifier", "role"]
    if attr_included_names is None:
        attr_included_names = edge_attr_data_names
    file_name = "data/preprocessed/graphs/protein_interactions/raw/protein_interactions_digraph.gpickle"
    G = nx.read_gpickle(file_name)
    G = add_name_rename(G)
    Gt = from_networkx(G)
    data = []
    for name in attr_included_names:
        data.append(Gt[name])
    data = np.array(data).T
    df = pd.DataFrame(data)
    ###
    if (enc == "ordinal") or (enc == "lookup"):
        edge_attr = df.to_numpy()
        enc_ordinal = OrdinalEncoder()
        enc_ordinal.fit(edge_attr)
        edge_attr = enc_ordinal.transform(edge_attr)
        edge_attr_tensor = torch.tensor(edge_attr)
        if enc == "lookup":
            full_feature_dims = []
            # Note on all ggi and nodes features need [categorical_names_input]
            for i in df:
                # Get feature dims list
                # reference https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
                full_feature_dims.append(df[i].value_counts().shape[0])
            enc_lookup = CategoricalEncoder(
                emb_dim=emb_dim, full_feature_dims=full_feature_dims
            )
            edge_attr_tensor = enc_lookup(edge_attr_tensor.long())
    elif enc == "onehot":
        # "None" string used previous for plotting
        df.replace("None", np.nan, inplace=True)
        edge_attr = df.to_numpy()
        # Not sparse to so can be converted to torch tensor
        enc = OneHotEncoder(sparse=False)
        enc.fit(edge_attr)
        edge_attr_tensor = torch.tensor(enc.transform(edge_attr))
    Gt.edge_attr = edge_attr_tensor
    ###
    edge_attr_drop_names = [
        "annotation_type",
        "detection_method_identifier",
        "role",
    ] + ["experiment_name"]
    # remove named features and keep one_hot features
    for name in edge_attr_drop_names:
        Gt[name] = None
    if save:
        torch.save(Gt, file_name.split(".")[0] + ".pt")
    return Gt


def nx_gene_interactions_to_torch(
    enc: str = "lookup",
    emb_dim: int = 2,
    attr_included_names: list = None,
    save=True,
):
    # One hot encoding protein interactions
    # default behavior is to include the all one_hot features inside conditional
    # names of features that could be potentially used
    edge_attr_data_names = [
        "interaction_details_annotation_type",
        # "interactions_detection_methods_identifier",
        # "interactions_details_phenotype",
        "interactions_details_role1",
        "p_value",
        "sgaScore",
    ]
    if attr_included_names is None:
        attr_included_names = edge_attr_data_names
    file_name = "data/preprocessed/graphs/gene_interactions/raw/gene_interactions_digraph.gpickle"
    G = nx.read_gpickle(file_name)
    G = add_name_rename(G)
    Gt = from_networkx(G)
    data = []
    for name in attr_included_names:
        data.append(Gt[name])
    data = np.array(data).T
    df = pd.DataFrame(data)
    # # "None" string used previous for plotting... this graph shouldn't have any nan values
    df.replace("None", np.nan, inplace=True)
    ## only different part compared to other graph functions.
    df.columns = edge_attr_data_names
    # replace with floats with median values
    df["sgaScore"] = df["sgaScore"].replace(np.nan, df["sgaScore"].median())
    df["p_value"] = df["p_value"].replace(np.nan, df["p_value"].median())
    # type case
    df = df.astype({"p_value": "float", "sgaScore": "float"})
    numeric_names = ["sgaScore", "p_value"]
    numeric_names_input = list(
        set(numeric_names).intersection(set(edge_attr_data_names))
    )
    edge_attr_numeric = df[numeric_names_input].to_numpy()
    edge_attr_numeric = torch.tensor(edge_attr_numeric)
    categorical_names = [
        "interaction_details_annotation_type",
        "interactions_detection_methods_identifier",
        "interactions_details_phenotype",
        "interactions_details_role1",
    ]
    categorical_names_input = list(
        set(categorical_names).intersection(set(edge_attr_data_names))
    )
    ###
    if (enc == "ordinal") or (enc == "lookup"):
        edge_attr_categorical = df[categorical_names_input].to_numpy()
        enc_ordinal = OrdinalEncoder()
        enc_ordinal.fit(edge_attr_categorical)
        edge_attr_categorical = enc_ordinal.transform(edge_attr_categorical)
        edge_attr_categorical_tensor = torch.tensor(edge_attr_categorical)
        if enc == "lookup":
            full_feature_dims = []
            for i in df[categorical_names_input]:
                # Get feature dims list
                # reference https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
                full_feature_dims.append(df[i].value_counts().shape[0])
            enc_lookup = CategoricalEncoder(
                emb_dim=emb_dim, full_feature_dims=full_feature_dims
            )
            edge_attr_categorical_tensor = enc_lookup(
                edge_attr_categorical_tensor.long()
            )
    elif enc == "onehot":
        # "None" string used previous for plotting
        df.replace("None", np.nan, inplace=True)
        edge_attr_categorical = (
            df[categorical_names_input].replace(np.nan, "None").to_numpy()
        )
        # Not sparse to so can be converted to torch tensor
        enc = OneHotEncoder(sparse=False)
        enc.fit(edge_attr_categorical)
        edge_attr_categorical_tensor = torch.tensor(
            enc.transform(edge_attr_categorical)
        )

    # # Not sparse to so can be converted to torch tensor
    # enc = OneHotEncoder(sparse=False)
    # enc.fit(edge_attr_categorical)
    # edge_attr_categorical = torch.tensor(enc.transform(edge_attr_categorical))
    ###
    # combine one_hot and numeric
    edge_attr_tensor = torch.concat(
        (edge_attr_categorical_tensor, edge_attr_numeric), dim=1
    )
    # concatenate tensors
    Gt.edge_attr = edge_attr_tensor
    edge_attr_drop_names = [
        "interaction_details_annotation_type",
        "interactions_detection_methods_identifier",
        "interactions_details_phenotype",
        "interactions_details_role1",
        "p_value",
        "sgaScore",
    ] + ["experiment_name"]
    # remove named features and keep one_hot features
    for name in edge_attr_drop_names:
        Gt[name] = None
    if save:
        torch.save(Gt, file_name.split(".")[0] + ".pt")
    return Gt


def nx_attr_to_torch(
    graph: str = None,
    enc: str = "lookup",
    emb_dim: int = 2,
    attr_included_names: list = None,
    save=True,
):
    print(f"Converting {graph} to torch Data:")
    if graph == "regulators":
        data = nx_regulators_to_torch(
            enc=enc,
            emb_dim=emb_dim,
            attr_included_names=attr_included_names,
            save=save,
        )
    elif graph == "protein_interactions":
        data = nx_protein_interactions_to_torch(
            enc=enc,
            emb_dim=emb_dim,
            attr_included_names=attr_included_names,
            save=save,
        )
    elif graph == "gene_interactions":
        data = nx_gene_interactions_to_torch(
            enc=enc,
            emb_dim=emb_dim,
            attr_included_names=attr_included_names,
            save=save,
        )
    elif graph == "yeastmine_node":
        data = node_reprs_to_data(
            enc=enc,
            emb_dim=emb_dim,
            attr_included_names=attr_included_names,
            save=save,
        )
    else:
        raise ValueError("graph not found")
    return data


def main():
    # synthetic_lethal = get_synthetic_lethal()
    # print(len(synthetic_lethal))
    # gene_node_dict()
    # # joining node features and edge features into one data object
    # data_reg = nx_attr_to_torch(graph="regulators", enc="lookup", emb_dim=2)
    # print(data_reg)
    # data_ppi = nx_attr_to_torch(graph="protein_interactions", enc="lookup", emb_dim=2)
    # print(data_ppi)
    # data_ggi = nx_attr_to_torch(graph="gene_interactions", enc="lookup", emb_dim=2)
    # print(data_ggi)
    data_yeastmine_node = nx_attr_to_torch(
        graph="yeastmine_node", enc="lookup", emb_dim=2
    )
    ############
    file_name_torch = "data/preprocessed/gene_reprs/yeastmine/node_reprs.pt"
    # load file_name_torch
    data = torch.load(file_name_torch)
    data.x
    #############3
    # print(data_yeastmine_node)
    # data_reg_full = join_edge_node_reprs(data_reg, data_yeastmine_node)
    # print(f"combined_node_attrs:{data_reg_full}")
    # # joining node features from yeastmine and unirep
    # data_unirep = unirep_to_data()
    # print(f"unirep_data:\n{data_unirep}")
    # if data_unirep.gene_name == data_yeastmine_node.gene_name:
    #     x_all = torch.concat((data_yeastmine_node.x, data_unirep.x), dim=1)
    #     print("yeastmine + unirep node features size:", x_all.shape)


if __name__ == "__main__":
    main()

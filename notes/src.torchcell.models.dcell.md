---
id: euw7ks1ua3afvqcu9bwb7uh
title: Dcell
desc: ''
updated: 1700278395667
created: 1694555579561
---

## Dcell model

- 2,526 GO terms define the `DCell` subsystems

### Dcell model - Input Training Data

$D=\left\{\left(X_1, y_1\right),\left(X_2\right.\right.$, $\left.\left.y_2\right), \ldots,\left(X_N, y_N\right)\right\}, (N - \text{sample number})$

$\forall i, X_i \in \mathbb{R}^M, X_i \in \{0,1\}, (0 = \text{wild type}$; $1=\text{disrupted})$

$y_i \in \mathbb{R}, (\text{relative growth rate, genetic interaction value})$

$t$ - subsystem

### Dimensionality Analysis

$$
\begin{aligned}
W^{(0)} & \in \mathbb{R}^{L_O^{(0)} \times L_I^{(0)}} \\
L_O^{(0)} & =\max (20,\lceil 0.3 \times 15\rceil) \\
& =\max (20,\lceil 4.5\rceil) \\
& =\max (20,5) \\
& =20 \\
L_I^{(0)} & =2 \\
W^{(0)} & \in \mathbb{R}^{20 \times 2}
\end{aligned}
$$

![](./assets/images/src.torchcell.models.dcell.md.pytorch-tanh.png)

[torch.nn.Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)

[torch.nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

## Dcell Model Drawio

![](assets/drawio/Dcell.drawio.png)

## Lua Not Compatible with Delta Redhat

```bash
mjvolk3@dt-login02 torch % lsb_release -a                                                                                                                                             13:56
LSB Version: :core-4.1-amd64:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch
Distributor ID: RedHatEnterprise
Description: Red Hat Enterprise Linux release 8.8 (Ootpa)
Release: 8.8
Codename: Ootpa
mjvolk3@dt-login02 torch % bash install-deps                                                                                                                                          13:56
==> Only Ubuntu, elementary OS, Fedora, Archlinux, OpenSUSE, Debian, CentOS and KDE neon distributions are supported.
```

## Subsetting GO by Date

Look into how we can subset GO by date. From the looks of this is not possible with the `gff`, but this data does exist in SGD. Just showing one term here.  We would have to cross reference with this data to get the GO subset.

```json
"go_details": [
    {
        "id": 6389520,
        "annotation_type": "manually curated",
        "date_created": "2002-11-26",
        "qualifier": "enables",
        "locus": {
            "display_name": "YDR210W",
            "link": "/locus/S000002618",
            "id": 1266542,
            "format_name": "YDR210W"
        },
        "go": {
            "display_name": "molecular function",
            "link": "/go/GO:0003674",
            "go_id": "GO:0003674",
            "go_aspect": "molecular function",
            "id": 290848
        },
        "reference": {
            "display_name": "SGD (2002)",
            "link": "/reference/S000069584",
            "pubmed_id": null
        },
        "source": {
            "display_name": "SGD"
        },
        "experiment": {
            "display_name": "ND",
            "link": "http://wiki.geneontology.org/index.php/No_biological_Data_available_(ND)_evidence_code"
        },
        "properties": []
    },
]
```

## Model Implementation - Passing Previous Subsystem Outputs

```mermaid
graph LR
    'GO:0000494'-leaf_node --> 'GO:0031126'
    'GO:0071051'-leaf_node --> 'GO:0031126'
```

```python
>>>len(G.nodes['GO:0031126']['gene_set'])
10
>>>len(G.nodes['GO:0000494']['gene_set'])
3
>>>len(G.nodes['GO:0071051']['gene_set'])
7
>>>dcell.subsystems['GO:0031126']
SubsystemModel(
  (linear): Linear(in_features=10, out_features=20, bias=True)
  (tanh): Tanh()
  (batchnorm): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
>>>dcell.subsystems['GO:0000494']
SubsystemModel(
  (linear): Linear(in_features=3, out_features=20, bias=True)
  (tanh): Tanh()
  (batchnorm): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
>>>dcell.subsystems['GO:0071051']
SubsystemModel(
  (linear): Linear(in_features=7, out_features=20, bias=True)
  (tanh): Tanh()
  (batchnorm): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

- The features from the two chid nodes `'GO:0000494'-leaf_node`,
`'GO:0071051'-leaf_node` should be concatenated with the boolean state vector of `'GO:0031126'`. This means that we should instead have size `50` coming in, the out features are still determined by the number of genes that are annotated to that node, so it will get maxed to 20.

```python
>>>dcell.subsystems['GO:0031126']
SubsystemModel(
  (linear): Linear(in_features=50, out_features=20, bias=True)
  (tanh): Tanh()
  (batchnorm): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

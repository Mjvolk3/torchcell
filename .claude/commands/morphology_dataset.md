# Adding Morphology Dataset

## Relevant Files For Reference

@/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/experiment_dataset.py
@/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/kuzmin2018.py
@/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/costanzo2016.py
@/Users/michaelvolk/Documents/projects/torchcell/torchcell/datamodels/schema.py
@/Users/michaelvolk/Documents/projects/torchcell/torchcell/knowledge_graphs/dataset_adapter_map.py
@/Users/michaelvolk/Documents/projects/torchcell/torchcell/adapters/costanzo2016_adapter.py
@/Users/michaelvolk/Documents/projects/torchcell/torchcell/adapters/cell_adapter.py
@/Users/michaelvolk/Documents/projects/torchcell/biocypher/config/torchcell_schema_config.yaml

## Morphology Dataset

We have already started by showing how the data can be downloaded from `scmdb`.

@/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/Ohya2005.py

## Dataset

Non-essential gene deletion mutants
The data sheets here have been published by Suzuki et al. (2018, BMC Genomics) by reanalysing the images first published in Ohya et al. (2005, PNAS) after a quality control.

Cell images of the mutants are available at SSBD:ssbd-repos-000349.

- 4718 gene deletion mutants
  - **Average data ( 27.7 MB )**
  - Number of cells for ratio parameter ( 885.6 KB )
  - Number of cells in specimen for ratio parameter (1.14 MB)
- 122 replicated wild-type (his3)
  - **Average data (1.06 MB)**
  - Number of cells for ratio parameter (23.4 KB)
  - Number of cells in specimen for ratio parameter (29.9 KB)

Bold average data is the only data that we need to download.

## Shared Data Storage

Since there is some instability of data download I am worried that it won't work if the user does not have mac with safari. For this reason we want to put the data onto a shared location and only try to download from source scmdb if the download from our shared location fails. For now just leave comments where we should put this if the first download attempt fails and then we will deal with it.

## WildType Data Download

This will serve as the reference for the individual mutant data. This case is a bit unique because in past there has only been one reference for the difference measurements that we have referred to but now there are many references, that is many wildtypes, for on mutant.  

[wildtype data](http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=wt122data.tsv)

## Tasks

The main task task is to create the `create_experiment` method of the dataset. This details how we represent data with `experiment`, `reference`, `publication` information. To do this we need to update @torchcell/torchcell/datamodels/schema.py to add for CALMORPH (name of software) experiment type, we Should Stick to typing so call it CalMorph. We also need associated CalMorph phenotype. This phenotype will be vector with the associated with the different labels from the CalMorph data. We also need to include the str labels... I think it is probably best to have a dictionary of the label, with float value. We can also put the strings for the or the abbreviated name to full string description map in the Phenotype class. the `schema.py` is the most sensitive file in the entire code base because it connects data import to database to downstream machine learning pipelines so we need to be very careful when adding or changing it. The next thing to do after that is to create an adapter so we can add the morphology dataset to the database. Prior to this we will need to add the necessary methods to the @/Users/michaelvolk/Documents/projects/torchcell/torchcell/adapters/cell_adapter.py to account for the data types. We will also have to add new phenotype to @/Users/michaelvolk/Documents/projects/torchcell/biocypher/config/torchcell_schema_config.yaml . After this is done then we should be able to edit @/Users/michaelvolk/Documents/projects/torchcell/biocypher/config/torchcell_schema_config.yaml and run this config @/Users/michaelvolk/Documents/projects/torchcell/torchcell/knowledge_graphs/create_kg.py to create the database with the new data added.

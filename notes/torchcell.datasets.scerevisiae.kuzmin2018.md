---
id: xfl1rdpnyjsb8ahz6g1kxxm
title: Kuzmin2018
desc: ''
updated: 1709757418439
created: 1705123822425
---
## Things We Know About Dmf Kuzmin

- It exists! This dataset is not just trigenic mutants, it has over 400,000 double mutants. But are they really? And that's where the trouble begins. Technically these are triple mutants but since the ho deletion in considered inert... or something like that, the triple mutants with one ho deletion are considered double mutants.
- One reference temperature as no temperature column data

## Unsure of Origin of Alleles

These alleles don't have much of an explanation, all other alleles are deletions signified by `Δ`.

```python
df[~df['Query allele name_1'].str.contains("Δ")]['Query allele name_1'].unique()
array(['ALG14-ph', 'rfc5-1', 'SPP381-ph', 'cdc7-4', 'uso1-1', 'cop1-1',
       'DAD1-ph', 'sec7-1', 'GPI19-PH', 'smc1-259', 'cdc4-1', 'sec53-6',
       'ret2-1', 'erg26-1', 'spc105-15', 'sec27-1', 'cdc43-2', 'swc4-4',
       'okp1-5', 'BIG1-ph', 'tid3-1', 'pan1-4', 'cdc35-1', 'sds22-5',
       'cof1-5', 'spc3-4', 'PWP1-ph', 'stt4-4', 'kap95-E126K', 'smc6-9',
       'gab1-2', 'ero1-1', 'tap42-11', 'arp9-1', 'RNA14-ph', 'lcb1-5',
       'smc5-6', 'sgt1-3', 'ufe1-1', 'ala1-1', 'kre5-ts2', 'sec16-2',
       'cdc60-ts', 'spn1-K192N'], dtype=object)
# Length is 44
```

```python
df[~df['Query allele name_2'].str.contains("Δ")]['Query allele name_2'].unique()
array(['PRE7-ph', 'sec31-1', 'brl1-3231', 'sam35-2', 'cdc23-1',
       'GPI16-PH', 'phs1-1', 'ask1-3', 'sec59-ts', 'SPC24-9', 'SPC98-PH'],
      dtype=object)
```

## All Double Mutants Contain the ho deletion

This is where we must face the ideal vs reality and choose. In the mood of a German idealist. While it betrays reality a bit to represent the triple deletions as double deletions, and the quadruple deletions and triple deletions as they are so labeled, it goes with the understanding and representation presented by the authors. We trust that the authors know the important points to emphasize, basically a deferral to experts justification for our idealism.

```python
(df['Query allele name'].str.contains("hoΔ")).sum()
410399
len(df)
410399
```

Annoyingly the `hoΔ` can be on the left hand or right hand side of the `+`.

## Processing Kuzmin Double Mutants in Trigenic Rows

I didn't know if the double mutants reported in the trigenic rows were new mutant data or from the double mutants in the previous rows. I thought that it would be redundant to add the double mutant information form the query so it could be possible that the double mutant data was new. After processing all rows we get a bunch of redundant data.

I used this to investigate the duplicates. It is not efficient to run this during creation of the dataset, so I just use it to find that the double mutant data is duplicated in the trigenic rows.

```python
>>> df["md5"] = df.apply(
       lambda row: hashlib.md5(
              json.dumps(
              self.create_experiment(row, self.reference_phenotype_std)[
                     0
              ].model_dump()
              ).encode("utf-8")
       ).hexdigest(),
       axis=1,
)
>>> print(df['md5'].value_counts()>1).sum()
0 # ??? digenic and trigenic 
0 # digenic only

>>> (df['md5'].value_counts()==1).sum()
410399 # digenic and trigenic
410399 # digenic only
```

We could then get rid of this code block.

```python
elif row["Combined mutant type"] == "trigenic":
       # Query 1
       if "KanMX_deletion" in row["query_perturbation_type_1"]:
              genotype.append(
              DeletionGenotype(
                     perturbation=SgaKanMxDeletionPerturbation(
                     systematic_gene_name=row["Query systematic name_1"],
                     perturbed_gene_name=row["Query allele name_1"],
                     strain_id=row["Query strain ID"],
                     )
              )
              )
       elif "allele" in row["query_perturbation_type_1"]:
              genotype.append(
              BaseGenotype(
                     perturbation=SgdAllelePerturbation(
                     systematic_gene_name=row["Query systematic name_1"],
                     perturbed_gene_name=row["Query allele name_1"],
                     strain_id=row["Query strain ID"],
                     )
              )
              )
       # Query 2
       if "KanMX_deletion" in row["query_perturbation_type_2"]:
              genotype.append(
              DeletionGenotype(
                     perturbation=SgaKanMxDeletionPerturbation(
                     systematic_gene_name=row["Query systematic name_2"],
                     perturbed_gene_name=row["Query allele name_2"],
                     strain_id=row["Query strain ID"],
                     )
              )
              )
       elif "allele" in row["query_perturbation_type_2"]:
              genotype.append(
              BaseGenotype(
                     perturbation=SgdAllelePerturbation(
                     systematic_gene_name=row["Query systematic name_2"],
                     perturbed_gene_name=row["Query allele name_2"],
                     strain_id=row["Query strain ID"],
                     )
              )
       )
```

## SmfKuzmin2018 Docker Import Issues with None and Special Characters

`SmfKuzmin2018` has stds don't have values so I was putting `None` which is somewhere converted to nan and the import fails. Another issue I fixed was that the `'` and the `Δ` symbols cannot be imported so I replaced them with `'` with `_prime` and `Δ` with `_delta`.  It is unclear what to do about the missing stds if we cannot import `None.` I thought this should just be mapped to null.

When I replace the `None` with some arbitrary double it works.

## Always Close the lmdb before pickling

```python
print(dataset.env)
> Environment() # some lmdb environment
# ⛔️ Cannot at this point be pickled
dataset.close_lmdb
print(dataset.env)
> None
# ✅ Can be pickled
```

First thing while debugging was that we could not pickle due to an object called `Environment`, naturally I assumed this was the `Environment` object related to data that I wrote. Instead it was the `lmdb` `Environment` object. This object cannot be pickled and it is created whenever data from the `lmdb` is accessed. Prior to pickling it must be made None. The `lmdb` can then be initialized after the dataset object is deserialized on whichever process. A sneaky one for sure 😈.

## Running Docker Interactive Will Merge Stdout and Return Value

This first script results in the print statements to be merged with the return value making bash_script_path_cleaned more like a log. And it prevents us from getting the proper string to call the bash script.

```bash
bash_script_path_cleaned=$(docker exec -it tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small)
```

The string is wrong Error ⛔️, we can see it is going line by line and trying to run bash on it. The `'INFO'` and `'This'` are output from biocypher.

```bash
----------------NOW BUILDING GRAPH---------------------
chmod: cannot access 'INFO': No such file or directory
chmod: cannot access 'This': No such file or directory
chmod: cannot access 'is': No such file or directory
chmod: cannot access 'BioCypher': No such file or directory
chmod: cannot access 'v0.5.37.'$'\r': No such file or directory
/bin/bash: line 2: biocypher-log/biocypher-20240221-063654.log: Permission denied
/bin/bash: line 2: INFO: command not found
/bin/bash: line 3: frozen: No such file or directory
/bin/bash: line 3: this: command not found
/bin/bash: line 4: ?,: No such file or directory
/bin/bash: line 5: ?,: No such file or directory
/bin/bash: line 6: INFO: command not found
/bin/bash: line 7: INFO:biocypher:Loading: command not found
/bin/bash: line 8: INFO: command not found
/bin/bash: line 9: INFO:biocypher:Instantiat
```

This is the **correct** command.

```bash
bash_script_path_cleaned=$(docker exec tc-neo4j python -m torchcell.knowledge_graphs.create_scerevisiae_kg_small)
```

## 2024.03.06 Why the Dataset methods Come after Super

Not sure where to put this note but I noticed that child classes of `ExperimentDataset` must always put their attributes and methods before `super()` because this is what runs process. If there are any attributes defined in the dataset that are needed in process they won't be in the scope of process.

---
id: dxhoxruso0jc7offn2jytqh
title: Kuzmin2018
desc: ''
updated: 1705681459925
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

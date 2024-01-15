---
id: dxhoxruso0jc7offn2jytqh
title: Kuzmin2018
desc: ''
updated: 1705136422447
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

## Single Mutant of Query and arra
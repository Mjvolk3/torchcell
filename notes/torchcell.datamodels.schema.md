---
id: 2qlur6rz1mzpmtltyf7w0j0
title: Ontology_pydantic
desc: ''
updated: 1706740152369
created: 1705045346511
---
## Costanzo Smf and Dmf Whiteboard Recap

- We don't want specific info in genotype lke array position to allow for joining genotypes
  - But without it, it is difficult to make the data come together properly
- We need a way to validate the `dmf` single mutant data to make sure we are referencing the correct mutant from the correct array
- I had some ideas on pulling `strain_id` information via another method, but I avoid this and just added the data to more refined pydantic models.
  - We want these more refined pydantic models because they allow us to take advantage of the authors preprocessing of their own data. Authors like to use some conventions, and this is evident in the follow up `Kuzmin` work. It is best to add these specific details to the data so they can be used for processing, but they need to easily removed so the more general data can be more easily merged.

![](./assets/drawio/ontology_pydantic_hourglass_data_model.png)

## Issues with the Current Data Scheme that Uses Different Named Phenotypes

If we look at the [pydantic data scheme](./assets/images/pydantic_data_scheme-2024.01.31.png) we it looks like the different genotypes are not playing any role. since they don't carry any additional attributes. I think this layer can likely be removed simplifying our data model. Also should rename this file to something like schema.

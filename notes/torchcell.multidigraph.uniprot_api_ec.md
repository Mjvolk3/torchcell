---
id: dukj36ne67p1lhd2e2igl0q
title: Uniprot_api_ec
desc: ''
updated: 1695246782606
created: 1695246628163
---

## GFF File Does not Contain EC Number

Gene Banker files ``.gbff` contain EC_number infomration which could be useful but the `gffutils` package is more useful for querying genes since it it built on an sqlite db. Its information is better organized and easier to parse. If we want things like EC Number they can be queried from different APIs. We can use the UniProt API to query EC number.

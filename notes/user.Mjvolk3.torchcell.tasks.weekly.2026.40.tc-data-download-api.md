---
id: 4zfer7uao6q95wos2rdq1zf
title: tc-data-download-api
desc: ''
updated: 1790716991414
created: 1790716991414
---

## 2026.09.29

- [x] T6 of [[plan.data-release-program.2026.09.29]] (issue #469): the `tc-data` download API [[torchcell.datasets.server]] with Swagger, the LMDB packager [[scripts.package_dataset_lmdb]] (deterministic tar.xz, `index.json` + `SHA256SUMS`, refuses unmanifested or stale builds; `smf_costanzo2016` packages to 1.05 MB in 5.1 s), [[torchcell.datasets.artifact]] records, the [[torchcell.datasets.client]] (resume + sha256 verify), the loader download path in [[torchcell.data.experiment_dataset]], shared hashed keys in [[torchcell.api_keys]], `Dockerfile.tc-data` + `docker-compose.tc-data.yml`, the Radiant recipe [[database.tc-data-endpoint]], and `docs/source/guide/downloads.md`; 65 new hermetic tests

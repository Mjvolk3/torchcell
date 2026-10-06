---
id: 9kqsnstakbuvwgw74t7xr6v
title: test-suite-buildout-p21
desc: ''
updated: 1791270631275
created: 1791270631275
---

## 2026.10.06

- [x] PR-21 of [[plan.test-suite-buildout.2026.09.25]]: the remaining reached code outside demo mains: the Sameith and Kemmeren download and parallel paths with the loader remainders ([[tests.torchcell.datasets.scerevisiae.test_sameith2015_synthetic]], [[tests.torchcell.datasets.scerevisiae.test_kemmeren2014_synthetic]], [[tests.torchcell.data.test_experiment_dataset]]), the NSA attention stack and the losses ([[tests.torchcell.nn.test_masked_attention_block]], [[tests.torchcell.nn.test_hetero_nsa]], [[tests.torchcell.nn.test_nsa_encoder]], [[tests.torchcell.losses.test_isomorphic_cell_loss]], [[tests.torchcell.losses.test_diffusion_loss]], [[tests.torchcell.losses.test_losses_dango]], [[tests.torchcell.datamodules.test_lazy_collate]]), the embedding wrappers and small models ([[tests.torchcell.models.test_models_esm2]], [[tests.torchcell.models.test_models_protT5]], [[tests.torchcell.models.test_mlp]], [[tests.torchcell.datasets.test_one_hot_gene]], [[tests.torchcell.data.test_hetero_data]]), every adapter's conf loading ([[tests.torchcell.adapters._adapter_init_harness]]), the PubChem curation tool and the literature helpers ([[tests.torchcell.datamodels.test_compound_identity_curate]], [[tests.torchcell.literature.test_extract]], [[tests.torchcell.literature.test_si_data]], [[tests.torchcell.literature.test_retrieve]], [[tests.torchcell.metabolism.test_betaxanthin]], [[tests.torchcell.sga.test_image]]); 405 new test functions, 55 pinned findings, two independent Opus 5.5 audits (418 reviewed, 1 rejected, 7 rewritten), TOTAL 65.1% to 68.1%; record in [[test-campaign.2026.09.25]]

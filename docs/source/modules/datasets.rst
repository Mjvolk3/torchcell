torchcell.datasets
==================

.. module:: torchcell.datasets

.. currentmodule:: torchcell.datasets

Dataset loaders. The top level of ``torchcell.datasets`` holds the per-gene embedding datasets (sequence language-model embeddings, codon frequencies, one-hot and random baselines), which derive from the in-memory embedding base in ``torchcell.data.embedding``, and ``NodeEmbeddingBuilder``, which builds them from a configuration. The experiment datasets live in ``torchcell.datasets.scerevisiae``: one module per source publication, each reading that publication's released data, converting its records into :mod:`torchcell.datamodels` experiments and storing them in LMDB. A class decorated with :func:`~torchcell.datasets.dataset_registry.register_dataset` is added to ``dataset_registry``, the name-to-class map that ``knowledge_graphs.create_kg`` and ``knowledge_graphs.kg_manifest`` look datasets up in; the table below lists every registered class.

.. contents:: Contents
    :local:

Embedding datasets
------------------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   NucleotideTransformerDataset
   FungalUpDownTransformerDataset
   CodonFrequencyDataset
   OneHotGeneDataset
   ProtT5Dataset
   GraphEmbeddingDataset
   Esm2Dataset
   CalmDataset
   RandomEmbeddingDataset
   NodeEmbeddingBuilder

Dataset registry
----------------

.. currentmodule:: torchcell.datasets.dataset_registry

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   register_dataset

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   dataset_registry

Registered *S. cerevisiae* experiment datasets
----------------------------------------------

The 52 classes in ``dataset_registry``, grouped by source module (one module per publication).

.. currentmodule:: torchcell.datasets.scerevisiae

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   auesukaree2009.EnvChemgenAuesukaree2009Dataset
   baryshnikova2010.SmfBaryshnikova2010Dataset
   bloom2019.Bloom2019Dataset
   cachera2023.BetaxanthinCachera2023Dataset
   caudal2024.CaudalPanTranscriptome2024Dataset
   cooper2010.AminoAcidCooper2010Dataset
   costanzo2016.DmfCostanzo2016Dataset
   costanzo2016.DmiCostanzo2016Dataset
   costanzo2016.SmfCostanzo2016Dataset
   costanzo2021.EnvChemgenCostanzo2021Dataset
   dasilveira2014.MetaboliteDaSilveira2014Dataset
   hillenmeyer2008.HetHillenmeyer2008Dataset
   hillenmeyer2008.HomHillenmeyer2008Dataset
   hoepfner2014.EnvChemgenHoepfner2014Dataset
   kemmeren2014.MicroarrayKemmeren2014Dataset
   kuzmin2018.DmfKuzmin2018Dataset
   kuzmin2018.DmiKuzmin2018Dataset
   kuzmin2018.SmfKuzmin2018Dataset
   kuzmin2018.TmfKuzmin2018Dataset
   kuzmin2018.TmiKuzmin2018Dataset
   kuzmin2020.DmfKuzmin2020Dataset
   kuzmin2020.DmiKuzmin2020Dataset
   kuzmin2020.SmfKuzmin2020Dataset
   kuzmin2020.TmfKuzmin2020Dataset
   kuzmin2020.TmiKuzmin2020Dataset
   lian2019.CrisprMagicLian2019Dataset
   lopez2024.IsobutanolScreenLopez2024Dataset
   lopez2024.IsobutanolValidatedLopez2024Dataset
   messner2023.ProteomeMessner2023Dataset
   mormino2022.CrispriMormino2022Dataset
   mota2024.EnvChemgenMota2024Dataset
   mulleder2016.AminoAcidMulleder2016Dataset
   nadal_ribelles2025.NadalRibellesPerturbSeq2025Dataset
   oduibhir2014.SmfODuibhir2014Dataset
   ohnuki2018.ScmdOhnuki2018Dataset
   ohnuki2022.ScmdOhnuki2022Dataset
   ohya2005.ScmdOhya2005Dataset
   ozaydin2013.CarotenoidOzaydin2013Dataset
   sameith2015.DmMicroarraySameith2015Dataset
   sameith2015.SmMicroarraySameith2015Dataset
   sgd.GeneEssentialitySgdDataset
   smith2006.FattyAcidSmith2006Dataset
   smith2016.CrispriChemgenSmith2016Dataset
   synth_leth_db.SynthLethalityYeastSynthLethDbDataset
   synth_leth_db.SynthRescueYeastSynthLethDbDataset
   vanacloig2022.EnvChemgenVanacloig2022Dataset
   wildenhain2015.EnvChemgenWildenhain2015Dataset
   xue2025.FattyAcidXue2025Dataset
   yeastphenome.YeastPhenomeDataset
   yoshida2012.OrganicAcidYoshida2012Dataset
   zelezniak2018.MetaboliteZelezniak2018Dataset
   zelezniak2018.ProteomeZelezniak2018Dataset

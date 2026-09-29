torchcell.adapters
==================

.. module:: torchcell.adapters

.. currentmodule:: torchcell.adapters

BioCypher adapters that turn a built torchcell dataset into knowledge-graph nodes and edges. There is one adapter class per registered dataset (for example ``SmfCostanzo2016Adapter`` or ``BetaxanthinCachera2023Adapter``), and most inherit from :class:`~torchcell.adapters.CellAdapter`, which reads the dataset's LMDB records and yields ``BioCypherNode`` and ``BioCypherEdge`` objects through ``get_nodes`` and ``get_edges``. The knowledge-graph builders in :mod:`torchcell.knowledge_graphs` pair each dataset class with its adapter through ``dataset_adapter_map``.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   CellAdapter
   SmfCostanzo2016Adapter
   DmfCostanzo2016Adapter
   DmiCostanzo2016Adapter
   SmfKuzmin2018Adapter
   DmfKuzmin2018Adapter
   TmfKuzmin2018Adapter
   DmiKuzmin2018Adapter
   TmiKuzmin2018Adapter
   SmfKuzmin2020Adapter
   DmfKuzmin2020Adapter
   TmfKuzmin2020Adapter
   DmiKuzmin2020Adapter
   TmiKuzmin2020Adapter
   GeneEssentialitySgdAdapter
   SynthLethalityYeastSynthLethDbAdapter
   SynthRescueYeastSynthLethDbAdapter
   ScmdOhya2005Adapter
   SmfODuibhir2014Adapter
   MicroarrayKemmeren2014Adapter
   SmMicroarraySameith2015Adapter
   DmMicroarraySameith2015Adapter
   CaudalPanTranscriptome2024Adapter
   NadalRibellesPerturbSeq2025Adapter
   ScmdOhnuki2018Adapter
   ScmdOhnuki2022Adapter
   CarotenoidOzaydin2013Adapter
   BetaxanthinCachera2023Adapter
   MetaboliteDaSilveira2014Adapter
   OrganicAcidYoshida2012Adapter
   IsobutanolScreenLopez2024Adapter
   IsobutanolValidatedLopez2024Adapter
   FattyAcidXue2025Adapter
   MetaboliteZelezniak2018Adapter
   ProteomeZelezniak2018Adapter
   ProteomeMessner2023Adapter
   AminoAcidMulleder2016Adapter
   AminoAcidCooper2010Adapter
   Bloom2019Adapter
   EnvChemgenCostanzo2021Adapter
   EnvChemgenAuesukaree2009Adapter
   EnvChemgenMota2024Adapter
   Smith2006Adapter
   Smith2016Adapter
   Lian2019Adapter
   Mormino2022Adapter
   EnvChemgenVanacloig2022Adapter
   EnvChemgenWildenhain2015Adapter
   EnvChemgenHoepfner2014Adapter
   HetHillenmeyer2008Adapter
   HomHillenmeyer2008Adapter
   SmfBaryshnikova2010Adapter

# torchcell/datasets/scerevisiae/yeastphenome
# [[torchcell.datasets.scerevisiae.yeastphenome]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/yeastphenome
# Test file: tests/torchcell/datasets/scerevisiae/test_yeastphenome.py
"""YeastPhenome (Turco & Baryshnikova 2023) curated growth-phenome loader.

Consumes the CURATED YeastPhenome data library -- ``yp-data`` (Zenodo
10.5281/zenodo.7714347), the compendium of Turco et al. 2023 (Sci Adv,
doi:10.1126/sciadv.adg5702, citation_key ``turcoGlobalAnalysisYeast2023``) -- as a
SECONDARY source. We ingest what YeastPhenome PRESENTS, not the primary papers, and
mark every field the curation does not carry as a typed ``ProvenanceGap`` rather than
guessing it.

ONE loader, MANY screens (NOT one loader per PMID). We consume a single pre-processed
intermediate, so each retained screen is a member of ``SCREENS`` and its source PMID is
per-record provenance (``Publication.pubmed_id``), not a separate loader file. Scope is
the GROWTH phenome only (~53% of YeastPhenome): a deletion x condition -> growth.
HAPLOID (BY4741/BY4742), HOMOZYGOUS-DIPLOID (BY4743) and HETEROZYGOUS-DIPLOID columns
all qualify (2026.10.12, #1045): the first two are the same total-absence
``KanMxDeletionPerturbation`` with only ``ReferenceGenome.ploidy`` differing, and a
``het`` column is one copy dropped 2 -> 1 in BY4743, the ``EngineeredCopyNumberPerturbation``
the Ohnuki 2018 loader uses. A column whose collection the curation marks as ambiguous
("hap ?", "hap a/hap alpha") stores the deletion with the strain a typed gap on the
reference; today's pinned screens carry none. Expression screens (~42%, Kemmeren) and
the ~5% mosaic (proteome/metabolome/morphology/localization/pH) are OUT (different
phenotype families, several already built directly from primaries); already-built
primaries (Costanzo/Kemmeren/Kuzmin/...) are EXCLUDED to avoid double-counting.

CONDITIONS (2026.10.12, #1045: every growth column with a representable condition is
stored; the previous build stored only a single dosed agent in a library medium). A
condition string is a comma-joined list of components, parsed by ``_parse_conditions``:
``standard`` is the base condition with no edit; ``temperature [N C]`` sets the
environment temperature; ``time [N h]`` / ``time [N gen]`` set ``duration_hours`` /
``duration_generations``; ``oxygen [0%]`` sets ``aerobicity="anaerobic"``; a dosed agent
``<name> [<value> <unit>]`` is a ``SmallMoleculePerturbation`` (the four plant defensins
are ``BiologicPerturbation``s); a repeated identical component (``sodium
chloride [0.6 M], time [48 h], sodium chloride [0.6 M]``) counts once. A component the
schema cannot carry makes the column ``condition_not_representable`` and it is counted,
never guessed: a dose RANGE (``hydrogen peroxide [1-1.2 mM]``, no single value and no
basis), an agent with NO dose (``gentamicin``, ``Mospilan``; ``Concentration`` needs a
value or a basis), a unit ``ConcentrationUnit`` lacks (``ng/ml``, ``U/ml``), and a
genotype edit written as a condition (``expression of human CYP1A2``, ``loss +
cytoduction of mtDNA``). Each such column is listed with its record count in the
dataset note; the units are an additive enum change for the schema branch.

MEDIA (2026.10.15, lane X of the ontology unification: the token's composition is typed, not spelled into a name).
The medium token is stored as released -- the WHOLE token, verbatim, in ``Media.name``
-- and everything it states about composition is read into typed fields by
``_parse_media_token``: a named dropout (``SC - Ade``) into ``dropouts``, an appended
component or a vehicle or a selection agent (``SC, HEPES [25 mM]``, ``SC + DMSO
[0.5%]``, ``YPD + G418 [200 ug/ml]``) into ``components`` at its stated dose, and a
nutrient CLASS the curation declares absent without naming the compound (``SC (-C)``,
``SD + His + Leu + Lys + Ura, -N``) into ``Media.dropout_roles``. Before this the name
was ``token.split("+")[0]``, so the ``+`` half reached no field at all (six tokens
stored the plain name ``SC`` over 212,402 records, six stored ``YPD`` over 239,081,
five stored ``SD`` over 23,354) and the retained half's statement sat inside a name
``media_identity`` drops: 53,951 records shared an environment node with a record
measured in a different medium, among them Galardini 2019's nitrogen-free arm and its
own control. An unknown token form RAISES rather than storing a medium with its
statement unread. The base RECIPE stays unexpanded: no library object is substituted
for the token, ``base_medium`` carries the join to the shared family, and a base the
library tables name is rich or synthetic by that table, any other base (``SC - Ura``,
``YPG [3%]``) is classified by its prefix (``YP`` rich;
``SC``/``SD``/``YNB``/``CSM``/``MM``/``SynBase`` synthetic) and anything else stores
``is_synthetic`` as a typed gap; a header with no medium field stores a ``Media`` whose
name, state and synthetic flag are typed gaps (the SGD precedent).

LABEL = the NPV (normalized phenotypic value). YeastPhenome's canonical label
(Turco2023) is a per-screen, mode-referenced modified z-score: the mode of each screen
(the most typical mutant ~ wild type) is the 0-reference and values are standardized
deviations (SD units; ``|NPV| > 3`` = strong). This is what YeastPhenome PRESENTS and it
fits ``EnvironmentResponsePhenotype`` (``measurement_type=z_score``, reference 0). We
read the NPV from ``<stem>_valuez.txt``; the original native-unit values in
``<stem>_value.txt`` (the pre-transform input) are NOT the label and stay retrievable at
the v1.0 pin if ever needed. YeastPhenome runs the same condition by different assays
(microarray vs barseq, ...) as SEPARATE screens, so the readout METHOD is recorded in
``units`` -- it is part of a record's identity (distinguishes near-replicate screens).
The EXACT modified-z formula is deferred to Turco2023 note S4 + yeastphenome.org
(offline) -> documented in ``units`` (a normalization-method gap).

PROVENANCE. Source bytes are pinned to the ``v1.0`` git tag of ``yeastphenome/yp-data``
(commit ``83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8``) -- the immutable freeze that
Zenodo 7714347 archives -- NOT the moving ``master`` branch, and each retrieved file is
verified against its pinned sha256. ``retrieval_method`` = ``zenodo``.

WHAT CURATION DOES NOT CARRY -> typed ``ProvenanceGap`` (never guessed): the environment
``temperature`` (absent from the curated condition metadata) and, on the phenotype,
``n_samples`` / ``environment_response_uncertainty`` / ``sample_unit`` (the replicate
design is not carried). All are ``not_carried_by_curation`` and anchored by ``looked_in``
= the Zenodo record. A future per-paper SI comb would upgrade these to sourced values.
"""

import json
import logging
import os
import os.path as osp
import pickle
import re
import urllib.request
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

import lmdb
from pydantic import BaseModel

from torchcell.data import (
    ExperimentDataset,
    Organism,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    BiologicAgentClass,
    BiologicPerturbation,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    EngineeredCopyNumberPerturbation,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    GenePerturbationType,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification import (
    Provenance,
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Immutable pin: the v1.0 tag of yeastphenome/yp-data = the freeze Zenodo 7714347
# archives. NOT `master` (which has drifted past the freeze).
YP_PIN = "83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8"
YP_RAW = f"https://raw.githubusercontent.com/yeastphenome/yp-data/{YP_PIN}/Datasets"

# The curation layer we consult -- anchors both the value provenance and every gap.
CURATION = Provenance(
    source_uri="https://doi.org/10.5281/zenodo.7714347",
    citation_key="turcoGlobalAnalysisYeast2023",
    method=(
        "curated YeastPhenome NPV (mode-referenced modified z-score) from yp-data "
        f"@ v1.0 ({YP_PIN[:12]}); exact modified-z formula deferred to Turco2023 note S4"
    ),
)
NPV_UNITS = (
    "YeastPhenome NPV: mode-referenced modified z-score (SD units; |NPV|>3 strong); "
    "exact modified-z formula deferred to Turco2023 note S4 / yeastphenome.org (offline)"
)

# Systematic ORF (R64) -- YeastPhenome value files are already ORF-resolved.
_SYSTEMATIC_RE = re.compile(r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4})$")
# One dosed component: "<name> [<value> <unit>]"; a value that is a range ("1-1.2")
# does not match and is reported as not representable.
_DOSE_RE = re.compile(
    r"^(?P<name>[^\[\]]+?)\s*\[(?P<val>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>[^\]]+)\]$"
)
_TEMPERATURE_RE = re.compile(r"^temperature \[(?P<val>[0-9]+(?:\.[0-9]+)?) C\]$")
_TIME_RE = re.compile(r"^time \[(?P<val>[0-9]+(?:\.[0-9]+)?) (?P<unit>h|gen)\]$")
_OXYGEN_RE = re.compile(r"^oxygen \[(?P<val>[0-9]+(?:\.[0-9]+)?)%\]$")

_UNIT_MAP = {
    "M": ConcentrationUnit.molar,
    "mM": ConcentrationUnit.millimolar,
    "uM": ConcentrationUnit.micromolar,
    "µM": ConcentrationUnit.micromolar,
    "nM": ConcentrationUnit.nanomolar,
    "ug/ml": ConcentrationUnit.ug_per_ml,
    "ug/mL": ConcentrationUnit.ug_per_ml,
    "g/L": ConcentrationUnit.g_per_l,
    "g/l": ConcentrationUnit.g_per_l,
    "%": ConcentrationUnit.percent,
    "% v/v": ConcentrationUnit.percent_v_v,
    "% w/v": ConcentrationUnit.percent_w_v,
}

# Plant-defensin conditions (PMID 31451498): sequence-identified antimicrobial
# PEPTIDES, not small molecules -- their identity is a sequence, not an InChIKey, so a
# ``Compound``/``SmallMoleculePerturbation`` cannot represent them. Retyped to
# ``BiologicPerturbation(agent_class=peptide)`` (audit DEFECT fix).
_DEFENSIN_NAMES = {"dmamp1", "nad1", "nbd6", "sbi6"}

# Rich (non-synthetic) base media; synthetic-defined media are is_synthetic=True. A base
# in neither table is classified by its prefix; anything else gaps ``is_synthetic``.
_RICH_MEDIA = {"YPD", "YPAD", "YEPD", "YP", "YPG", "YPGal", "YP4D"}
_SYNTHETIC_MEDIA = {"SD", "SC", "YNB", "CSM", "MM", "SynBase"}
_RICH_PREFIXES = ("YP",)
_SYNTHETIC_PREFIXES = ("SC", "SD", "YNB", "CSM", "MM", "SynBase")

# Deletion-collection backgrounds. Ploidy lives on ``ReferenceGenome``, so a HAPLOID
# deletion and a HOMOZYGOUS-DIPLOID deletion are the same total-absence
# ``DeletionPerturbation`` -- only the background differs. A HETEROZYGOUS diploid
# column (``het``) is one copy dropped 2 -> 1 in BY4743: ``EngineeredCopyNumberPerturbation``
# with ``copy_number=1``, ``reference_copy_number=2`` (the Ohnuki 2018 leaf). A zygosity
# token outside this table (an ambiguous or mixed collection the curation writes as
# "hap ?" or "hap a/hap alpha") is stored with the strain as a typed gap.
_ZYGOSITY: dict[str, tuple[str, Literal["haploid", "diploid"]]] = {
    "hom": ("BY4743", "diploid"),
    "het": ("BY4743", "diploid"),
    "hap a": ("BY4741", "haploid"),
    "hap alpha": ("BY4742", "haploid"),
    "hap a (post-SGA)": ("BY4741-derived (post-SGA)", "haploid"),
    "hap alpha (post-SGA)": ("BY4742-derived (post-SGA)", "haploid"),
}

# The retained GROWTH screens (v1.0 pin), each NOT already built as a torchcell primary.
# ONE loader, MANY screens: the loader parses EVERY data column and keeps the ones it can
# encode as a single dosed-compound growth condition in a representable background
# (drop-and-log the rest). Widen by appending {pmid, stem, valuez_sha256}.
# Skip list (already-built primaries, excluded up front): costanzo 27708008/33958448,
# kemmeren 24766815, kuzmin 29674565/32586993, baryshnikova 21076421, auesukaree 19638689,
# hoepfner 24360837, wildenhain 27136353, hillenmeyer 18420932, smith 16738555/26956608,
# ohya 16365294, ozaydin 22918085, vanacloig 35883225, mota 38419072, + the
# expression/metabolome/morphology primaries.
SCREENS: list[dict[str, str]] = [
    {
        "pmid": "12543677",
        "stem": "blackburn_avery_2003",  # 2 env; hap a
        "valuez_sha256": "264a4f2c6d8da3c66ea376f04554d7e950d825cd853da2a47c7439f3860ab149",
    },
    {
        "pmid": "16820484",
        "stem": "trotter_grant_2006",  # 1 env; hom
        "valuez_sha256": "95091c4fd0f8a525169444731a596fe462b10f358df5ce1a4b85fb2ab71f6d3b",
    },
    {
        "pmid": "17043098",
        "stem": "doostzadeh_langston_2007",  # 2 env; hom
        "valuez_sha256": "d24e7d31956b3ccf4194f4d40ab8dde2bacf8afbc481a6c7db0338ba237ff8dd",
    },
    {
        "pmid": "17630978",
        "stem": "pagani_arino_2007",  # 1 env; hap a
        "valuez_sha256": "91a06a26f6bd289c8157a6f0bc2ad0295af639e6ef07320e4793cb9a5954a037",
    },
    {
        "pmid": "19416971",
        "stem": "khozoie_avery_2009",  # 1 env; hom
        "valuez_sha256": "99eb77d2743aaebc2bde69802b65ce7b9d2b8994799bea2057b43728340c1b48",
    },
    {
        "pmid": "20206679",
        "stem": "zhao_jiang_2010",  # 1 env; hom
        "valuez_sha256": "65434146ce1c146f36fac65f59b83d2c71f6c0a2fcc9fd4998e4345e8b0bc9a9",
    },
    {
        "pmid": "20960220",
        "stem": "jayakody_kitagaki_2011",  # 1 env; hom
        "valuez_sha256": "02bf3e296437c95a75a104b743bd744131a17627db7df0717ddeb18f133fc7d4",
    },
    {
        "pmid": "22102822",
        "stem": "berry_gasch_2011",  # 7 env; hom
        "valuez_sha256": "a776b817f6062e1746a19a622180878efca15b5c262e06b30de300347fee98f8",
    },
    {
        "pmid": "22271309",
        "stem": "yibmantasiri_bellows_2012",  # 1 env; hom
        "valuez_sha256": "35dee547d8509942b194c7a813c2ffd7db59341b793884dcd7cbceae6d344adf",
    },
    {
        "pmid": "22384333",
        "stem": "hoon_nislow_2011",  # 3 env; hom
        "valuez_sha256": "13167918d8201140283f2ac63f0bba6ea43fc4302634709c5081b40296a0e1f1",
    },
    {
        "pmid": "23335509",
        "stem": "cuesta_marban_mollinedo_2013",  # 1 env; hap a
        "valuez_sha256": "d3ec6b9a55e7a3976828b2f24e2320df0598517a708e145e815a3ddfcaade60e",
    },
    {
        "pmid": "23689276",
        "stem": "bowie_fyles_2013",  # 2 env; hap a
        "valuez_sha256": "06b75af1c0a1deb5295a689f3a1679a838045db7fd897b1574b3c56e1cdbaf32",
    },
    {
        "pmid": "23733464",
        "stem": "islahudin_avery_2013",  # 1 env; hom
        "valuez_sha256": "c9d1b2546fbb856944afe469a0b72d6dd846d1b347083f1a86db3fab9e6cdca6",
    },
    {
        "pmid": "23832094",
        "stem": "tun_wu_2013",  # 2 env; hom
        "valuez_sha256": "570545476ee593f05f824413c2b81faeb3835994e9dd3300ee32219f92c23518",
    },
    {
        "pmid": "23874964",
        "stem": "bojsen_regenberg_2013",  # 1 env; hap a
        "valuez_sha256": "6afee16837b9aa63460cb03d6533bc0d2c8d56091be82981d89edcd23da719fd",
    },
    {
        "pmid": "23915247",
        "stem": "porcu_ragnini_wilson_2013",  # 1 env; hap a
        "valuez_sha256": "43ee6b592077e0db2bbea7788adca070b4ee9f18c968483d3089b68e15022255",
    },
    {
        "pmid": "24035500",
        "stem": "shimada_gasser_2013",  # 1 env; hom
        "valuez_sha256": "1b72dcaff7baf677dddaf2e00d80e174754e82bac89e441389b520c5590d5d2b",
    },
    {
        "pmid": "24262822",
        "stem": "mattiazziusaj_petrovic_2014",  # 2 env; hap a
        "valuez_sha256": "06868ce4d3cc75a82f47c216d017f9b0c2df7342058c1c6a7c818edadf24c578",
    },
    {
        "pmid": "24926745",
        "stem": "tun_wu_2014",  # 2 env; hom
        "valuez_sha256": "9906ee3323490a59df3fedfe3975f92ddac2b684879ebccdf4cc78db08cccbbc",
    },
    {
        "pmid": "25074250",
        "stem": "hwang_naganuma_2014",  # 1 env; hap alpha
        "valuez_sha256": "39491c903471f92ec1f4d87c939ecdb54aff499191c662edabb944574bf7fd98",
    },
    {
        "pmid": "25519239",
        "stem": "aiyar_steinmetz_2014",  # 1 env; hom
        "valuez_sha256": "d7b3d3602c49b143d5ddada609db708569d56e6a18e91adb453632f415e9059b",
    },
    {
        "pmid": "25773006",
        "stem": "du_jiang_2015",  # 1 env; hom
        "valuez_sha256": "2e417a8b0bf61c970040e2a301b802ce9d1f1faa8e50878bada51e6917423e89",
    },
    {
        "pmid": "26341223",
        "stem": "garcia_arroyo_2015",  # 1 env; hap a
        "valuez_sha256": "7a9ed3a3a3ce17c38daa25c45b2be4d99d305707fc7fd525f2867b989551f674",
    },
    {
        "pmid": "26357016",
        "stem": "frohlich_walther_2015",  # 1 env; hap a
        "valuez_sha256": "3533ab847099e5249811c8f56ad21271fbafc9e4c0429f07f08bad80366c7e9d",
    },
    {
        "pmid": "26994103",
        "stem": "luo_jiang_2016",  # 2 env; hom
        "valuez_sha256": "58b2a2f615f66427081e58741ce01d719aef6239a0137a26ddcfed233707f3d9",
    },
    {
        "pmid": "28076367",
        "stem": "bozaquel_morais_montero_lomeli_2017",  # 2 env; hap a
        "valuez_sha256": "704ff2c36a9f05225020f0c215365010b9206b5dbb518d3b6c4e285d6df0dc38",
    },
    {
        "pmid": "28472365",
        "stem": "maclean_zhang_2017",  # 3 env; hom
        "valuez_sha256": "2e0b57b509198b5727839e85b59d6196303c778de65dc31d385f8f344c00b29f",
    },
    {
        "pmid": "28592509",
        "stem": "acton_giaever_2017",  # 3 env; hom
        "valuez_sha256": "3e2f8850fba50118820d6866f507bf78669a2fc80a94c79ceab03627e1c4759e",
    },
    {
        "pmid": "28743744",
        "stem": "nomura_inoue_2017",  # 1 env; hap a
        "valuez_sha256": "f4c088bc636e9ec14872d81a758d53231d1003a1d29a2bd421d2acf6b61ed4a7",
    },
    {
        "pmid": "28973557",
        "stem": "delarosa_vulpe_2017",  # 3 env; hom
        "valuez_sha256": "f97243313a645eb4b1dfdff6c085a79869bca9c3732aa562ec409d4a974d6373",
    },
    {
        "pmid": "30233513",
        "stem": "grosjean_blaudez_2018",  # 1 env; hap a
        "valuez_sha256": "c3948d89e5a03353ca0ad45947edfb9e21000dee8565e34b69cde7eda61ac2b8",
    },
    {
        "pmid": "30381188",
        "stem": "ruta_farcasanu_2018",  # 1 env; hap a
        "valuez_sha256": "2b4458bae938cb19cec05f3841954651eba66ceace7321c83c5f136283e5c6bb",
    },
    {
        "pmid": "30647105",
        "stem": "alhoch_tang_2019",  # 2 env; hap a
        "valuez_sha256": "a191e3cbe715df8c54a2a4fd2d58f6d2d7d49e336b5f3fcc44aeaeff953602c8",
    },
    {
        "pmid": "31427087",
        "stem": "zhao_han_2019",  # 2 env; hap a
        "valuez_sha256": "98fc190594d4b941acd19e719ac7704a9f9067b686d2eea184ef74ea933248c8",
    },
    {
        "pmid": "31451498",
        "stem": "parisi_bleackley_2019",  # 4 env; hap a
        "valuez_sha256": "d64f6b172212be20a188196e519047ab13be8c7402b901029976a07a2c928a40",
    },
    {
        "pmid": "31885205",
        "stem": "galardini_beltrao_2019",  # 8 env; hap a
        "valuez_sha256": "d293951be28e1ddd84b2581d288e78ea3b0febb1b03c9261d4b0b99ee0d042ee",
    },
    {
        "pmid": "31904504",
        "stem": "zhao_deng_2020",  # 1 env; hom
        "valuez_sha256": "8a835af18d622c4a29ee42bf0d72a1db2e01ca7f3e96019df7db8b4fbf8200b5",
    },
    {
        "pmid": "32391394",
        "stem": "wilcox_austriaco_2020",  # 1 env; hap alpha
        "valuez_sha256": "ed53a4e6688d2d2ac0d2674f6dbe510e45be9f8776c41176288e4f559026da16",
    },
    {
        "pmid": "32658971",
        "stem": "helsen_jelier_2020",  # 1 env; hap a
        "valuez_sha256": "cbd9baceac7bbafd5542f760612dfcec5817ef0864d39189e04c6b5365e14b59",
    },
    {
        "pmid": "32904421",
        "stem": "stenger_westermann_2020",  # 1 env; hap a
        "valuez_sha256": "5a2a96d1b4c35377daffae01c3c20b8b5b001cc15e024db22c57d345c8148728",
    },
    {
        "pmid": "32994210",
        "stem": "stjohn_fasullo_2020",  # 1 env; hom
        "valuez_sha256": "57dac87728f9227bae0f7e2c7eb98eef849f5c22b3dbcee46a6d631fffa43c6a",
    },
    {
        "pmid": "33082270",
        "stem": "berg_brandl_2020",  # 1 env; hap a
        "valuez_sha256": "b2fa4053834a4a1b8148db3e7c991d4d2917d048810e945c9b82ba9e5a6397d1",
    },
    {
        "pmid": "33690632",
        "stem": "nicastro_devirgilio_2021",  # 1 env; hap a
        "valuez_sha256": "1d1cda620c51cf59a50ce09d963bcaf9f625a10b5da72ba1940c494ba9c5d936",
    },
    {
        "pmid": "33924665",
        "stem": "jin_liu_2021",  # 1 env; hap a
        "valuez_sha256": "3dc1c65dfea5ef7be3a2788f0a51750fc5d4262a6a95f3a99ef0ec13b38ca063",
    },
    {
        "pmid": "34071169",
        "stem": "kipanga_luyten_2021",  # 2 env; hap alpha
        "valuez_sha256": "a89a68ab6dcc5bfb05a727ca8a5aa7f2ef670481e217ef4b1706416c82fc629e",
    },
    {
        "pmid": "34944020",
        "stem": "jin_liu_2021",  # 1 env; hap a
        "valuez_sha256": "8c6e8a68976ef3bd2af1720c1a2659d3f907a5a5211f2fb6f756e996085646a2",
    },
    {
        "pmid": "35495664",
        "stem": "cao_liu_2022",  # 1 env; hap a
        "valuez_sha256": "590b872331439342ab09c27fd9da9e534239540418d44a508abc4e2962b5d290",
    },
]


_METHOD_RE = re.compile(r"\((?P<method>[^)]+)\)")


def _gap(field: str) -> ProvenanceGap:
    """A ``not_carried_by_curation`` gap anchored to the YeastPhenome curation layer."""
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_carried_by_curation,
        looked_in=CURATION,
    )


def _readout_method(phenotype: str) -> str:
    """The readout modality inside the phenotype string, e.g. ``growth (culture turbidity)``
    -> ``culture turbidity``. YeastPhenome runs the SAME condition by different assays
    (microarray vs barseq, ...) as SEPARATE screens; the method is part of a record's
    identity (it distinguishes near-replicate screens), so it is recorded in ``units``.
    """
    m = _METHOD_RE.search(phenotype)
    return m.group("method").strip() if m else phenotype.strip()


def _column_meta(col_header: str) -> dict[str, str | None] | None:
    """Split a column header ``<hom|het> | <phenotype> | <condition> | [<media> |] <author>``.

    The author is ALWAYS the last ``|`` field; media is the second-to-last ONLY when the
    header has 5 fields (some screens omit media -> 4 fields, media absent). Returns None
    for a malformed (<4-field) header.
    """
    parts = [p.strip() for p in col_header.split("|")]
    if len(parts) < 4:
        return None
    return {
        "zygosity": parts[0],
        "phenotype": parts[1],
        "condition": parts[2],
        "media": parts[3] if len(parts) >= 5 else None,
    }


class ParsedCondition(BaseModel):
    """What a curated condition string says about the environment.

    ``unrepresentable`` lists the components no schema field can carry (a dose range,
    an undosed agent, a unit the enum lacks, a genotype edit written as a condition);
    a column with any is counted under ``condition_not_representable``, never stored
    with a guessed environment.
    """

    perturbations: list[EnvironmentPerturbationType] = []
    temperature: Temperature | None = None
    duration_hours: float | None = None
    duration_generations: float | None = None
    aerobicity: str = "aerobic"
    unrepresentable: list[str] = []


def _parse_agent(
    component: str,
) -> SmallMoleculePerturbation | BiologicPerturbation | None:
    """One dosed agent ``<name> [<value> <unit>]``; None when the schema cannot carry it.

    The four plant defensins (PMID 31451498) route to a peptide ``BiologicPerturbation``
    before the small-molecule path; every other agent is a resolver-routed
    ``SmallMoleculePerturbation`` (fill-or-gap on the compound identity). A dose range,
    a missing dose, or a unit outside ``_UNIT_MAP`` returns None.
    """
    m = _DOSE_RE.match(component)
    if m is None:
        return None
    unit = _UNIT_MAP.get(m.group("unit").strip())
    if unit is None:
        return None
    name = m.group("name").strip()
    concentration = Concentration(value=float(m.group("val")), unit=unit)
    if name.lower() in _DEFENSIN_NAMES:
        return BiologicPerturbation(
            agent_class=BiologicAgentClass.peptide,
            name=name,
            concentration=concentration,
        )
    return SmallMoleculePerturbation(
        compound=resolved_compound(name), concentration=concentration
    )


def _parse_conditions(condition: str) -> ParsedCondition:
    """Parse a curated condition string into its typed environment edits.

    Components are comma-separated (``", "``); an identical repeated component counts
    once. ``standard`` is the base condition; ``temperature [N C]``, ``time [N h|gen]``
    and ``oxygen [0%]`` set the environment's own fields; anything else is a dosed
    agent or is reported as unrepresentable.
    """
    parsed = ParsedCondition()
    seen: list[str] = []
    for raw in condition.split(", "):
        component = raw.strip()
        if not component or component in seen:
            continue
        seen.append(component)
        if component == "standard":
            continue
        if (m := _TEMPERATURE_RE.match(component)) is not None:
            parsed.temperature = Temperature(value=float(m.group("val")))
            continue
        if (m := _TIME_RE.match(component)) is not None:
            if m.group("unit") == "h":
                parsed.duration_hours = float(m.group("val"))
            else:
                parsed.duration_generations = float(m.group("val"))
            continue
        if (m := _OXYGEN_RE.match(component)) is not None:
            if float(m.group("val")) != 0.0:
                parsed.unrepresentable.append(component)
                continue
            parsed.aerobicity = "anaerobic"
            continue
        agent = _parse_agent(component)
        if agent is None:
            parsed.unrepresentable.append(component)
            continue
        parsed.perturbations.append(agent)
    return parsed


def _media_gap(field: str, looked_in_note: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_carried_by_curation,
        looked_in=CURATION,
        note=looked_in_note,
    )


def _base_medium(base: str) -> str | None:
    """The ``MEDIA_LIBRARY`` key a released medium token joins at, else None.

    The token's leading word (``SC`` of ``SC - Ade``, ``YPD`` of ``YPD``) when it is a
    library key; else the rich or synthetic prefix the token starts with when that
    prefix is one (``YP`` of ``YPG [3%]``). ``CSM`` joins at ``SC``: the library's SC
    object is the CSM + YNB + glucose formulation (its sourced components), so a medium
    the curation names by its supplement mixture is that family. The released name
    stays ``name``; the key is what makes the medium a derived member of the shared
    family rather than free text. No library object is substituted for the token.
    """
    head = re.split(r"[\s,\-(\[]", base, maxsplit=1)[0]
    if head == "CSM":
        return "SC"
    if head in MEDIA_LIBRARY:
        return head
    for prefix in (*_RICH_PREFIXES, *_SYNTHETIC_PREFIXES):
        if base.startswith(prefix) and prefix in MEDIA_LIBRARY:
            return prefix
    return None


# --- begin lane X: the released medium token's composition, in typed fields --------- #
# A curated media token is a base medium followed by ``+``-joined additions, any of
# which may carry ``,``-separated suffixes. Every form the v1.0 pin releases:
#
#   SC                                   the base alone
#   SC - Ade                             a NAMED dropout off the base
#   SC (-C)                              a ROLE-level dropout (no compound named)
#   SC, Glycerol [2%]                    an appended component at a stated dose
#   SC, HEPES [25 mM]                    the same, a buffer
#   YPG [3%]                             a dose on the base's own, unexpanded component
#   SC + DMSO [0.5%]                     a vehicle in the medium at a stated fraction
#   YPD + G418 [200 ug/ml] + DMSO [1%]   a selection agent and a vehicle
#   SD + his + leu + lys + ura           auxotroph supplements, no dose stated
#   SD + His + Leu + Lys + Ura, -N       the same, nitrogen-free
#   SD + His + Leu + Lys + Ura (2)       the curation's own index, which states no
#                                        composition and stays in the released name
#
# Before this the loader stored ``media_field.split("+")[0]`` as ``name`` and nothing
# else, so the ``+`` half of the token reached no field at all: six released tokens
# stored the plain name ``SC`` over 212,402 records, six stored ``YPD`` over 239,081 and
# five stored ``SD`` over 23,354 (474,837 of 524,682 records carried a name that is not
# the token). What the retained half states -- a dropout, a carbon source, a buffer --
# sat inside ``name``, which ``media_identity`` drops, so the five SC dropout arms and
# their complete control were ONE served environment node over 30,628 records, two
# vehicle controls and a buffered medium were one over 13,649, and a nitrogen-starved
# medium was the node of its own control over 9,674.
#
# Each compound the token names is read through the shared resolver, so a dropout joins
# the SAME compound entity another dataset's medium lists as a component. The
# curation's three-letter spellings are not resolver keys, so this table maps each to
# one, with the role it fills. A vehicle has no role of its own in
# ``MediaComponentRole`` and takes ``other``; a selection agent is a COMPONENT by the
# schema's own ruling (constant to the medium, not the studied edit).
_VEHICLE = "vehicle"
_SUPPLEMENT = "supplement"
_APPENDED = "appended"
_SELECTION = "selection"

_MEDIUM_COMPOUNDS: dict[str, tuple[str, MediaComponentRole, str]] = {
    "dmso": ("DMSO", MediaComponentRole.other, _VEHICLE),
    "meoh": ("methanol", MediaComponentRole.other, _VEHICLE),
    "etoh": ("ethanol", MediaComponentRole.other, _VEHICLE),
    "g418": ("G418 (geneticin)", MediaComponentRole.selection_agent, _SELECTION),
    "ade": ("adenine", MediaComponentRole.nucleobase, _SUPPLEMENT),
    "arg": ("L-arginine", MediaComponentRole.amino_acid, _SUPPLEMENT),
    "his": ("L-histidine", MediaComponentRole.amino_acid, _SUPPLEMENT),
    "leu": ("L-leucine", MediaComponentRole.amino_acid, _SUPPLEMENT),
    "lys": ("L-lysine", MediaComponentRole.amino_acid, _SUPPLEMENT),
    "met": ("L-methionine", MediaComponentRole.amino_acid, _SUPPLEMENT),
    "trp": ("L-tryptophan", MediaComponentRole.amino_acid, _SUPPLEMENT),
    "ura": ("uracil", MediaComponentRole.nucleobase, _SUPPLEMENT),
    "glycerol": ("glycerol", MediaComponentRole.carbon_source, _APPENDED),
    "hepes": ("HEPES", MediaComponentRole.buffer, _APPENDED),
}

#: What a component line means, by the position the token puts the compound in.
_COMPONENT_NOTES: dict[str, str] = {
    _VEHICLE: "the curation states this vehicle in the medium field of the column "
    "header, beside the base medium and not in the condition, so it is a component of "
    "the medium rather than the solvent slot of a dosed compound -- which is also the "
    "only encoding that holds for the two columns whose condition is 'standard' (a "
    "vehicle control has no perturbation to carry a Solvent)",
    _SUPPLEMENT: "a nutrient the curation adds to the base medium to complement the "
    "deletion collection's auxotrophies. CultureEnvironment.auxotroph_supplements is "
    "the slot for these, and this dataset's experiment class states the base "
    "Environment, which has none; the token states them as part of the medium, so that "
    "is where they are stored, with base_medium still carrying the join to the base",
    _APPENDED: "a component the curation appends to the base medium after a comma, "
    "with its dose",
    _SELECTION: "a selection agent is a COMPONENT, not an environment perturbation: it "
    "is constant to the medium rather than the studied edit (schema.py, Media section)",
}

#: A dose written on the base word itself names the base's own characteristic
#: component, which the curation abbreviates to a letter and never expands. Only the
#: bases whose abbreviated component's ROLE is unambiguous are read; any other dosed
#: base raises rather than guessing which ingredient the dose belongs to.
_BASE_DOSED_ROLE: dict[str, MediaComponentRole] = {
    "YPG": MediaComponentRole.carbon_source
}

#: The nutrient CLASS a token declares absent without naming the compound removed.
_ROLE_DROPOUTS: dict[str, MediaComponentRole] = {
    "C": MediaComponentRole.carbon_source,
    "N": MediaComponentRole.nitrogen_source,
}

# "<name> [<value> <unit>]" anywhere a medium token states a dose.
_MEDIA_DOSE_RE = re.compile(
    r"^(?P<name>.*\S)\s*\[(?P<val>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>[^\]]+)\]$"
)
# "SC (-C)": the base with a nutrient CLASS removed.
_ROLE_DROPOUT_RE = re.compile(r"^(?P<rest>.*\S)\s*\(-(?P<role>[A-Z])\)$")
# "SD + His + Leu + Lys + Ura (2)": the curation's own index for a second medium it
# spells identically; it states nothing about composition.
_CURATION_INDEX_RE = re.compile(r"^(?P<rest>.*\S)\s*\((?P<index>[0-9]+)\)$")
# "SC - Ade": the base with a NAMED nutrient dropped.
_NAMED_DROPOUT_RE = re.compile(r"^(?P<base>\S+)\s*-\s*(?P<nutrient>[A-Za-z]+)$")


class ParsedMediaToken(BaseModel):
    """The typed composition a curated media token states, and its base word."""

    base_word: str
    components: list[MediaComponent] = []
    dropouts: list[Compound] = []
    dropout_roles: list[MediaComponentRole] = []


def _media_dose(value: str, unit: str) -> Concentration:
    """A dose a medium token states; an unmapped unit raises rather than being dropped."""
    mapped = _UNIT_MAP.get(unit.strip())
    if mapped is None:
        raise ValueError(
            f"the curated medium token states a dose in {unit!r}, which "
            "ConcentrationUnit does not carry; add the unit rather than dropping the dose"
        )
    return Concentration(value=float(value), unit=mapped)


def _media_component(
    key: str, dose: Concentration | None, token: str
) -> MediaComponent:
    """One compound a medium token names, resolved and roled, with its stated dose."""
    entry = _MEDIUM_COMPOUNDS.get(key.lower())
    if entry is None:
        raise ValueError(
            f"the curated medium token {token!r} names {key!r}, which is in no "
            "_MEDIUM_COMPOUNDS row; add the compound, its role and its note rather "
            "than storing the token with the statement left inside its name"
        )
    resolver_key, role, kind = entry
    return MediaComponent(
        compound=resolved_compound(resolver_key),
        role=role,
        concentration=dose,
        note=_COMPONENT_NOTES[kind],
    )


def _parse_media_token(token: str) -> ParsedMediaToken:
    """Read a curated media token into typed composition, or raise on an unknown form.

    No fallback: a token form this grammar does not cover raises, because the
    alternative is a stored medium whose statement sits unparsed inside its name, which
    is the defect this function exists to remove. The ``SCREENS`` list is pinned by
    sha256, so the 28 released tokens are a closed set; widening it is what surfaces a
    new form.
    """
    subpieces: list[tuple[bool, str]] = []
    for piece_index, piece in enumerate(token.split("+")):
        for sub_index, sub in enumerate(piece.split(",")):
            text = sub.strip()
            if text:
                subpieces.append((piece_index == 0 and sub_index == 0, text))
    if not subpieces or not subpieces[0][0]:
        raise ValueError(f"the curated medium token {token!r} states no base medium")

    parsed = ParsedMediaToken(base_word="")
    for is_base, text in subpieces:
        rest = text
        if (m := _CURATION_INDEX_RE.match(rest)) is not None:
            rest = m.group("rest")
        if (m := _ROLE_DROPOUT_RE.match(rest)) is not None:
            role = _ROLE_DROPOUTS.get(m.group("role"))
            if role is None:
                raise ValueError(
                    f"the curated medium token {token!r} declares the nutrient class "
                    f"{m.group('role')!r} absent, which _ROLE_DROPOUTS does not name"
                )
            parsed.dropout_roles.append(role)
            rest = m.group("rest")
        elif rest in {f"-{letter}" for letter in _ROLE_DROPOUTS}:
            parsed.dropout_roles.append(_ROLE_DROPOUTS[rest[1:]])
            continue
        dose: Concentration | None = None
        if (m := _MEDIA_DOSE_RE.match(rest)) is not None:
            dose = _media_dose(m.group("val"), m.group("unit"))
            rest = m.group("name").strip()
        if not is_base:
            parsed.components.append(_media_component(rest, dose, token))
            continue
        if (m := _NAMED_DROPOUT_RE.match(rest)) is not None:
            entry = _MEDIUM_COMPOUNDS.get(m.group("nutrient").lower())
            if entry is None:
                raise ValueError(
                    f"the curated medium token {token!r} drops "
                    f"{m.group('nutrient')!r}, which is in no _MEDIUM_COMPOUNDS row"
                )
            parsed.dropouts.append(resolved_compound(entry[0]))
            rest = m.group("base")
        parsed.base_word = rest
        if dose is None:
            continue
        role = _BASE_DOSED_ROLE.get(rest)
        if role is None:
            raise ValueError(
                f"the curated medium token {token!r} doses the base {rest!r}, whose "
                "abbreviated component _BASE_DOSED_ROLE does not name"
            )
        parsed.components.append(
            MediaComponent(
                compound=Compound(
                    name=f"{rest} carbon source, abbreviated by the curation"
                ),
                role=role,
                concentration=dose,
                definition=ComponentDefinition.composition_deferred,
                note=f"the token {token!r} states the dose of the base's own "
                f"{role.value} and abbreviates that ingredient to a letter; "
                "YeastPhenome's curation carries no expansion, so the stated dose is "
                "kept against an unexpanded line rather than a guessed molecule",
            )
        )
    if not parsed.base_word:
        raise ValueError(f"the curated medium token {token!r} states no base medium")
    return parsed


def _media_statement(token: str, screen: dict[str, str]) -> SourcedValue:
    """The released token itself, pinned to the bytes of the screen file it came from.

    The token is a verbatim substring of that file's header line, and the header is the
    only place the curation states a medium, so one medium-level quote is the audit
    anchor for every typed field read out of it.
    """
    return SourcedValue(
        value=token,
        provenance=Provenance(
            source_uri=f"Datasets/{screen['pmid']}/{screen['stem']}_valuez.txt",
            citation_key=CURATION.citation_key,
            sha256=screen["valuez_sha256"],
        ),
        quote=token,
        note="the media field of the column header, verbatim; every typed component, "
        "dropout and dropped role on this medium is read from this one statement, and "
        "nothing else about the recipe is stated by the curation",
    )


def _parse_media(
    media_field: str | None, phenotype: str, screen: dict[str, str]
) -> Media:
    """The curated media token (``YPD + EtOH``) as a ``Media``, stored as released.

    ``name`` is the token VERBATIM; what the token states about composition is read
    into ``components``, ``dropouts`` and ``dropout_roles`` by
    :func:`_parse_media_token`, so no statement is left inside the name (lane X, 2026-10-15).
    ``state`` is inferred from the growth readout: colony-size / spot-assay /
    killing-zone => solid, else (turbidity / pooled / CFU) => liquid. ``is_synthetic``
    comes from the two library tables, else from the base word's prefix, else is a typed
    gap. A header with no medium field (4 fields) stores a ``Media`` whose name, state
    and synthetic flag are all typed gaps.

    The base recipe itself stays unexpanded: no library object is substituted for the
    token, so ``components`` holds exactly what the token states and ``base_medium``
    carries the join to the shared family, which is this loader's standing treatment of
    a secondary curation layer that names a medium without printing it.
    """
    if media_field is None:
        note = "the curated column header carries no medium field (4-field header)"
        return Media(
            name=None,
            state=None,
            is_synthetic=None,
            provenance_gaps=[
                _media_gap(field, note) for field in ("name", "state", "is_synthetic")
            ],
        )
    token = media_field.strip()
    parsed = _parse_media_token(token)
    base = parsed.base_word
    p = phenotype.lower()
    state = "solid" if ("colony" in p or "spot" in p or "killing" in p) else "liquid"
    shared: dict[str, Any] = {
        "name": token,
        "state": state,
        "base_medium": _base_medium(base),
        "components": parsed.components,
        "dropouts": parsed.dropouts,
        "dropout_roles": parsed.dropout_roles,
        "provenance": [_media_statement(token, screen)],
    }
    if base in _RICH_MEDIA or base.startswith(_RICH_PREFIXES):
        return Media(is_synthetic=False, **shared)
    if base in _SYNTHETIC_MEDIA or base.startswith(_SYNTHETIC_PREFIXES):
        return Media(is_synthetic=True, **shared)
    return Media(
        is_synthetic=None,
        provenance_gaps=[
            _media_gap(
                "is_synthetic",
                f"the curated medium token {token!r} names a base in neither "
                "the rich nor the synthetic table and with no classifying prefix",
            )
        ],
        **shared,
    )


# --- end lane X -------------------------------------------------------------------- #


@register_dataset
class YeastPhenomeDataset(ExperimentDataset):
    """Curated YeastPhenome growth-phenome env x geno -> NPV z-score screens."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/yeastphenome",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: curated rows carry systematic ORFs)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The curated NPV file for each retained screen (pinned). Label = the NPV; the
        original native-unit ``value.txt`` stays retrievable at the v1.0 pin if needed.
        """
        return [f"{s['pmid']}_{s['stem']}_valuez.txt" for s in SCREENS]

    def download(self) -> None:
        """Fetch each screen's NPV file from the v1.0 pin; verify its pinned sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for s in SCREENS:
            dest = osp.join(self.raw_dir, f"{s['pmid']}_{s['stem']}_valuez.txt")
            if osp.exists(dest):
                continue
            url = f"{YP_RAW}/{s['pmid']}/{s['stem']}_valuez.txt"
            log.info("YeastPhenome: downloading %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            write_verified(data, dest, s["valuez_sha256"], url)

    def _read_screen(self, s: dict[str, str]) -> tuple[list[str], list[list[str]]]:
        """Return (header columns, data rows as cell lists) for one screen's NPV file."""
        path = osp.join(self.raw_dir, f"{s['pmid']}_{s['stem']}_valuez.txt")
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
        return header, rows

    def _environment(self, media: Media, parsed: ParsedCondition) -> Environment:
        """The environment a condition string states; temperature is a typed gap when
        the condition does not set it (the curation carries no growth temperature).
        """
        return Environment(
            media=media,
            temperature=parsed.temperature,
            perturbations=parsed.perturbations,
            aerobicity=parsed.aerobicity,
            duration_hours=parsed.duration_hours,
            duration_generations=parsed.duration_generations,
            provenance_gaps=[]
            if parsed.temperature is not None
            else [_gap("temperature")],
        )

    @post_process
    def process(self) -> None:
        """Build env x geno -> NPV records for every GROWTH column of every screen.

        Each data COLUMN of a screen is one (phenotype x condition). Every growth column
        whose condition the schema can carry is stored; a column whose condition has an
        unrepresentable component, a non-growth column and a malformed header are
        counted, never guessed. A row id that is not ORF-shaped is stored as released.
        """
        verify_raw_files(
            self.raw_dir,
            {
                f"{s['pmid']}_{s['stem']}_valuez.txt": s["valuez_sha256"]
                for s in SCREENS
            },
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_env_kept = 0
        drops: Counter[str] = Counter()
        not_representable: list[dict[str, Any]] = []
        released_rows: set[str] = set()
        with env.begin(write=True) as txn:
            for s in SCREENS:
                header, rows = self._read_screen(s)
                pub_dump = Publication(
                    pubmed_id=s["pmid"],
                    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/",
                ).model_dump()
                kept_here = 0
                for col in range(1, len(header)):
                    if not header[col].strip():
                        continue
                    meta = _column_meta(header[col])
                    if meta is None:
                        drops["malformed_header"] += 1
                        continue
                    if "growth" not in (meta["phenotype"] or "").lower():
                        drops["not_growth"] += 1
                        continue
                    parsed = _parse_conditions(meta["condition"] or "")
                    n_cells = sum(
                        1 for cells in rows if col < len(cells) and cells[col].strip()
                    )
                    if parsed.unrepresentable:
                        drops["condition_not_representable"] += 1
                        not_representable.append(
                            {
                                "screen": s["stem"],
                                "pmid": s["pmid"],
                                "column": col,
                                "condition": meta["condition"],
                                "components": parsed.unrepresentable,
                                "n_records": n_cells,
                            }
                        )
                        log.info(
                            "YeastPhenome %s col %d: condition %r not representable "
                            "(%s); %d released values not stored",
                            s["stem"],
                            col,
                            meta["condition"],
                            parsed.unrepresentable,
                            n_cells,
                        )
                        continue
                    media = _parse_media(meta["media"], meta["phenotype"] or "", s)
                    zygosity = meta["zygosity"] or ""
                    background = _ZYGOSITY.get(zygosity)
                    # Readout method (microarray/barseq/turbidity/...) is part of the
                    # record identity -> recorded in units, so same-condition different-
                    # method near-replicate screens stay distinct (not false duplicates).
                    units = f"{NPV_UNITS} [readout: {_readout_method(meta['phenotype'] or '')}]"
                    environment = self._environment(media, parsed)
                    ref_dump = self._reference(
                        environment, units, background, zygosity
                    ).model_dump()
                    n_env_kept += 1
                    kept_here += 1
                    for cells in rows:
                        if col >= len(cells) or not cells[col].strip():
                            continue
                        orf = cells[0].strip()
                        if not _SYSTEMATIC_RE.match(orf):
                            # a row id with no ORF form goes on the released-gene-name
                            # leaf verbatim (#1045; 0 today), listed in the ledger
                            released_rows.add(orf)
                        experiment = self._experiment(
                            orf, float(cells[col]), environment, units, zygosity
                        )
                        txn.put(
                            f"{idx}".encode(),
                            pickle.dumps(
                                {
                                    "experiment": experiment.model_dump(),
                                    "reference": ref_dump,
                                    "publication": pub_dump,
                                }
                            ),
                        )
                        idx += 1
                log.info(
                    "YeastPhenome %s (PMID %s): %d encodable environment(s)",
                    s["stem"],
                    s["pmid"],
                    kept_here,
                )
        env.close()
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as fh:
            json.dump(
                {
                    "dataset": self.name,
                    "kept_records": idx,
                    "kept_environments": n_env_kept,
                    "columns_not_stored": dict(drops),
                    "condition_not_representable": not_representable,
                    "stored_on_released_gene_name_leaf": sorted(released_rows),
                },
                fh,
                indent=2,
            )
        log.info(
            "Wrote %d YeastPhenome records over %d environments (%d screens); "
            "dropped columns/rows: %s",
            idx,
            n_env_kept,
            len(SCREENS),
            dict(drops),
        )

    def _reference(
        self,
        environment: Environment,
        units: str,
        background: tuple[str, Literal["haploid", "diploid"]] | None,
        zygosity: str,
    ) -> EnvironmentResponseExperimentReference:
        """The mode (~ wild-type) baseline: NPV 0 in the same environment.

        ``strain``/``ploidy`` come from the screen's collection background (haploid BY4741 /
        BY4742 or diploid BY4743). A zygosity token outside ``_ZYGOSITY`` (an ambiguous
        or mixed collection) stores the token itself as the strain name, which is what
        the curation says of it.
        """
        strain, ploidy = background if background is not None else (zygosity, "haploid")
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=0.0,  # mode ~ WT -> 0 in the NPV space
            units=units,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain=strain,
                ploidy=ploidy,
                assembly_set=SGD_S288C_R64,
                assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    def _experiment(
        self, orf: str, npv: float, environment: Environment, units: str, zygosity: str
    ) -> EnvironmentResponseExperiment:
        """One deletion x condition -> NPV z-score experiment.

        A haploid or homozygous-diploid deletion is total absence, the
        ``KanMxDeletionPerturbation``; a heterozygous-diploid column (``het``) is one copy
        dropped 2 -> 1, the ``EngineeredCopyNumberPerturbation``. Curation gives only the
        ORF, so ``perturbed_gene_name`` is the ORF (no common name carried). A row id
        with no ORF form goes on ``ReleasedIdentifierPerturbation`` verbatim (#1045), kind
        ``heterozygous_deletion`` for a ``het`` column and ``deletion`` otherwise.
        """
        perturbation: GenePerturbationType
        if not _SYSTEMATIC_RE.match(orf):
            perturbation = ReleasedIdentifierPerturbation(
                systematic_gene_name=orf,
                gene_namespace="scerevisiae_s288c_sgd_systematic",
                perturbed_gene_name=orf,
                identifier_kind="row identifier of the curated matrix, as released",
                released_kind=(
                    ReleasedKind.heterozygous_deletion
                    if zygosity == "het"
                    else ReleasedKind.deletion
                ),
                cassette="KanMX",
            )
        elif zygosity == "het":
            perturbation = EngineeredCopyNumberPerturbation(
                systematic_gene_name=orf,
                perturbed_gene_name=orf,
                copy_number=1,
                reference_copy_number=2,
                marker="KanMX",
            )
        else:
            perturbation = KanMxDeletionPerturbation(
                systematic_gene_name=orf, perturbed_gene_name=orf
            )
        phenotype = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=npv,
            units=units,
            provenance_gaps=[
                _gap("n_samples"),
                _gap("environment_response_uncertainty"),
                _gap("sample_unit"),
            ],
        )
        return EnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=Genotype(perturbations=[perturbation]),
            environment=environment,
            phenotype=phenotype,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/yeastphenome")
    dataset = YeastPhenomeDataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

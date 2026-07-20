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
the GROWTH phenome only (~53% of YeastPhenome): a COMPLETE-loss-of-function single-ORF
deletion x chemical/stress condition -> growth. Both HAPLOID (BY4741/BY4742) and
HOMOZYGOUS-DIPLOID (BY4743) collections qualify -- the deletion is identical total
absence, only ``ReferenceGenome.ploidy`` differs (haploid BY4741 is in fact torchcell's
most common deletion background). HETEROZYGOUS diploid screens are OUT (gene dosage /
haploinsufficiency = a different perturbation type). Expression screens (~42%, Kemmeren)
and the ~5% mosaic (proteome/metabolome/morphology/localization/pH) are OUT (different
phenotype families, several already built directly from primaries); already-built
primaries (Costanzo/Kemmeren/Kuzmin/...) are EXCLUDED to avoid double-counting.

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

import hashlib
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

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    Publication,
    ReferenceGenome,
    SmallMoleculePerturbation,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.verification import Provenance, ProvenanceGap, ProvenanceGapReason

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
# One CLEAN dosed compound: "<name> [<value> <unit>]" with NO other '[' or ',' in the
# name (so a multi-component condition like "time [5 gen], X [1 uM]" is REJECTED, never
# mis-parsed -- it drops to the worklist instead).
_DOSE_RE = re.compile(
    r"^(?P<name>[^\[\],]+?)\s*\[(?P<val>[0-9.]+)\s*(?P<unit>[^\]]+)\]$"
)

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
}
# Rich (non-synthetic) base media; synthetic-defined media are is_synthetic=True.
_RICH_MEDIA = {"YPD", "YPAD", "YEPD", "YP", "YPG", "YPGal", "YP4D"}
_SYNTHETIC_MEDIA = {"SD", "SC", "YNB", "CSM", "MM", "SynBase"}

# Complete-loss-of-function backgrounds we can represent. Ploidy lives on
# ``ReferenceGenome``, so a HAPLOID deletion and a HOMOZYGOUS-DIPLOID deletion are the same
# total-absence ``DeletionPerturbation`` -- only the background differs. (Haploid BY4741 is
# in fact torchcell's most common deletion background.) EXCLUDED: ``het`` (heterozygous
# diploid = gene DOSAGE / haploinsufficiency, a different perturbation type needing
# ``EngineeredCopyNumberPerturbation``) and ambiguous/mixed collections ("hap ?",
# "hap a/hap alpha", ...), which are drop-and-logged.
_ZYGOSITY: dict[str, tuple[str, Literal["haploid", "diploid"]]] = {
    "hom": ("BY4743", "diploid"),
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


def _parse_condition(condition: str) -> SmallMoleculePerturbation | None:
    """Parse a CLEAN single dosed-compound condition; None if not encodable (-> drop-and-log).

    Handles ``<compound> [<value> <unit>]`` only. Physical stresses, dose ranges,
    multi-component conditions, and unmapped units return None and are dropped-and-logged --
    never guessed, never mis-parsed.
    """
    m = _DOSE_RE.match(condition.strip())
    if m is None:
        return None
    unit = _UNIT_MAP.get(m.group("unit").strip())
    if unit is None:
        return None
    return SmallMoleculePerturbation(
        compound=Compound(name=m.group("name").strip()),
        concentration=Concentration(value=float(m.group("val")), unit=unit),
    )


def _parse_media(media_field: str | None, phenotype: str) -> Media | None:
    """Parse the curated media token (``YPD + EtOH``) into a ``Media``; None if absent/unknown.

    Base = the token before '+'. ``state`` is inferred from the growth readout: colony-size /
    spot-assay / killing-zone => solid, else (turbidity / pooled / CFU) => liquid.
    """
    if media_field is None:
        return None
    base = media_field.split("+")[0].strip()
    if base in _RICH_MEDIA:
        is_synthetic = False
    elif base in _SYNTHETIC_MEDIA:
        is_synthetic = True
    else:
        return None
    p = phenotype.lower()
    state = "solid" if ("colony" in p or "spot" in p or "killing" in p) else "liquid"
    return Media(name=base, state=state, is_synthetic=is_synthetic)


@register_dataset
class YeastPhenomeDataset(ExperimentDataset):
    """Curated YeastPhenome growth-phenome env x geno -> NPV z-score screens."""

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
            digest = hashlib.sha256(data).hexdigest()
            if digest != s["valuez_sha256"]:
                raise RuntimeError(
                    f"{s['stem']} valuez sha256 mismatch: got {digest}, "
                    f"expected {s['valuez_sha256']}"
                )
            with open(dest, "wb") as handle:
                handle.write(data)

    def _read_screen(self, s: dict[str, str]) -> tuple[list[str], list[list[str]]]:
        """Return (header columns, data rows as cell lists) for one screen's NPV file."""
        path = osp.join(self.raw_dir, f"{s['pmid']}_{s['stem']}_valuez.txt")
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
        return header, rows

    def _environment(
        self, media: Media, perturbation: SmallMoleculePerturbation
    ) -> Environment:
        """Environment carrying one dosed compound; temperature is a typed gap."""
        return Environment(
            media=media,
            temperature=None,  # not carried by YeastPhenome curation -> gap below
            perturbations=[perturbation],
            provenance_gaps=[_gap("temperature")],
        )

    @post_process
    def process(self) -> None:
        """Build env x geno -> NPV records for every ENCODABLE column of every screen.

        Each data COLUMN of a screen is one (phenotype x condition). We keep columns that
        are homozygous + a growth readout + a single dosed-compound condition + a known
        base medium; every other column is dropped-and-logged (the worklist), never guessed.
        """
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_env_kept = 0
        drops: Counter[str] = Counter()
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
                    background = _ZYGOSITY.get(meta["zygosity"] or "")
                    if background is None:
                        # heterozygous (dosage) or an ambiguous/mixed collection
                        drops["het_or_ambiguous_zygosity"] += 1
                        continue
                    if "growth" not in (meta["phenotype"] or "").lower():
                        drops["not_growth"] += 1
                        continue
                    media = _parse_media(meta["media"], meta["phenotype"] or "")
                    if media is None:
                        drops["media_absent_or_unknown"] += 1
                        continue
                    pert = _parse_condition(meta["condition"] or "")
                    if pert is None:
                        drops["condition_unparseable"] += 1
                        log.info(
                            "YeastPhenome %s col %d: drop condition %r",
                            s["stem"],
                            col,
                            meta["condition"],
                        )
                        continue
                    # Readout method (microarray/barseq/turbidity/...) is part of the
                    # record identity -> recorded in units, so same-condition different-
                    # method near-replicate screens stay distinct (not false duplicates).
                    units = f"{NPV_UNITS} [readout: {_readout_method(meta['phenotype'] or '')}]"
                    strain, ploidy = background
                    environment = self._environment(media, pert)
                    ref_dump = self._reference(
                        environment, units, strain, ploidy
                    ).model_dump()
                    n_env_kept += 1
                    kept_here += 1
                    for cells in rows:
                        if col >= len(cells) or not cells[col].strip():
                            continue
                        orf = cells[0]
                        if not _SYSTEMATIC_RE.match(orf):
                            drops["non_orf_row"] += 1
                            continue
                        experiment = self._experiment(
                            orf, float(cells[col]), environment, units
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
        strain: str,
        ploidy: Literal["haploid", "diploid"],
    ) -> EnvironmentResponseExperimentReference:
        """The mode (~ wild-type) baseline: NPV 0 in the same environment.

        ``strain``/``ploidy`` come from the screen's collection background (haploid BY4741 /
        BY4742 or homozygous-diploid BY4743) -- the deletion itself is identical total
        absence either way.
        """
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=0.0,  # mode ~ WT -> 0 in the NPV space
            units=units,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain=strain, ploidy=ploidy
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    def _experiment(
        self, orf: str, npv: float, environment: Environment, units: str
    ) -> EnvironmentResponseExperiment:
        """One homozygous-diploid deletion x condition -> NPV z-score experiment."""
        # Homozygous diploid deletion = total absence -> DeletionPerturbation family;
        # ploidy (diploid) lives on the ReferenceGenome. Curation gives only the ORF,
        # so perturbed_gene_name = the ORF (no common name carried).
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
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
            genotype=genotype,
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

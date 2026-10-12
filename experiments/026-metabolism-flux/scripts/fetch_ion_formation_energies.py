# experiments/026-metabolism-flux/scripts/fetch_ion_formation_energies.py
# [[experiments.026-metabolism-flux.scripts.fetch_ion_formation_energies]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/fetch_ion_formation_energies.py

r"""Build a sha256-pinned standard formation energy table for monatomic aqueous ions.

**Why a separate table exists at all.** eQuilibrator's component contribution is a group
decomposition plus a reactant contribution, and a bare ion has neither: it has no groups to
decompose and it does not appear in the training reactions. Measured in
``thermo_tier_d_probe.json``: the cache holds a compound for every one of potassium, sodium,
calcium, zinc, copper and cadmium, each with a group vector, and ``standard_dg_formation``
returns nothing for all of them under every accession the GEM carries. No pKa predictor,
structure writer or decomposer changes that. What these species DO have is a standard
formation energy measured and evaluated in the thermochemical literature, so the remedy is a
table with provenance rather than a computation.

THE SOURCE
----------
CODATA Key Values for Thermodynamics (Cox, Wagman and Medvedev, 1989), the internationally
agreed evaluation. Retrieved as a PDF, mirrored under ``$DATA_ROOT`` with its sha256, and
PARSED rather than transcribed: every number this script uses is extracted from the mirrored
text and the verbatim line it came from is stored beside it.

The table gives the standard enthalpy of formation and the standard entropy at 298.15 K, not
the Gibbs energy, so the Gibbs energy is DERIVED here from the formation reaction, which is
written out rather than implied. For an ion :math:`M^{z}` with the aqueous proton as the
reference (CODATA lists ``H+ aq 0 0``, which is that convention),

.. math::

    M(\text{standard state}) + z\,\mathrm{H}^+(aq)
        \rightarrow M^{z}(aq) + \tfrac{z}{2}\,\mathrm{H}_2(g)

    \Delta_r H^\circ = \Delta_f H^\circ(M^{z})

    \Delta_r S^\circ = S^\circ(M^{z}) + \tfrac{z}{2} S^\circ(\mathrm{H}_2)
                       - S^\circ_{\text{elem, per atom}}

    \Delta_f G^\circ(T) = \Delta_r H^\circ - T \Delta_r S^\circ

with :math:`\Delta_f H^\circ` and :math:`S^\circ(\mathrm{H}^+,aq)` both zero by the same
convention. One expression covers cations and anions: for :math:`z<0` the
:math:`\tfrac{z}{2}\mathrm{H}_2` term moves to the left, which is the anion's formation
reaction, and the elemental entropy is taken per mole of ATOMS so a diatomic reference state
enters as half its tabulated value.

Carrying :math:`\Delta_f G^\circ(T) = \Delta_r H^\circ - T\Delta_r S^\circ` to a temperature
other than 298.15 K assumes :math:`\Delta_r S^\circ` is constant over the interval, which
over the 5 K from 298.15 to 303.15 is worth well under the tabulated uncertainties. Stated
rather than hidden.

THE DERIVATION IS CHECKED, NOT ASSERTED
---------------------------------------
Each derived value is compared against the independently published Gibbs energy for the same
ion from the NBS tables (Wagman et al. 1982), reproduced in the CRC Handbook. Those are a
different evaluation, so they are a CHECK and not the values used; the deviation is reported
per ion and the run fails if any exceeds a stated tolerance.

THE TRANSFORM
-------------
A bare ion has no dissociable protons and binds no magnesium in this treatment, so the
Legendre transform to biochemical conditions has no pH term and no magnesium term: only the
extended Debye-Huckel ionic-strength term survives, and it scales with :math:`z^2`. The
coefficients are eQuilibrator's own ``debye_hueckel``, and this script ASSERTS that the
transform it records equals ``equilibrator_cache.thermodynamic_constants._legendre_transform``
to machine precision, so the copy that :mod:`torchcell.metabolism.thermo_pipeline` carries
(which must import without eQuilibrator) is verified rather than restated.

**What the values do and do not change.** In a reaction that is elementally balanced in the
ion's element, the ion's absolute formation energy appears with coefficients summing to zero
and CANCELS exactly; the same is true of the ionic-strength term whenever the compartments
share an ionic strength. The absolute value therefore matters only for reactions that are
NOT elementally balanced, which in a GEM are the exchange, sink and biomass pseudo-reactions,
where a standard reaction energy has no physical meaning. The pipeline reports both facts.

Writes ``$DATA_ROOT/data/torchcell/ion_formation_energies/`` (the mirrored PDF, its text,
and ``ion_formation_energies.json``) plus a copy of the JSON in ``results/``.

Run under the eQuilibrator environment, which has the transform to check against::

    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/fetch_ion_formation_energies.py
"""

from __future__ import annotations

import hashlib
import os
import os.path as osp
import re
import subprocess
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
MIRROR_DIR = osp.join(DATA_ROOT, "data", "torchcell", "ion_formation_energies")
PDF_NAME = "codata-key-values-for-thermodynamics.pdf"
TEXT_NAME = "codata-key-values-for-thermodynamics.txt"
TABLE_NAME = "ion_formation_energies.json"

SOURCE_URL = (
    "https://www.science.co.il/chemistry/databases/"
    "Codata-key-values-for-thermodynamics.pdf"
)
RETRIEVAL_METHOD = "direct_url"
RETRIEVAL_COMMAND = f"curl -sSL -o {PDF_NAME} '{SOURCE_URL}'"
EXTRACTION_COMMAND = f"pdftotext -layout {PDF_NAME} {TEXT_NAME}"
CITATION = (
    "Cox, J. D., Wagman, D. D. and Medvedev, V. A. (1989). CODATA Key Values for "
    "Thermodynamics. Hemisphere Publishing Corporation, New York."
)

#: The reference temperature the source tabulates at.
SOURCE_TEMPERATURE_K = 298.15

#: The conditions the 026 build prices at. The table stores the untransformed value; the
#: pipeline stage applies the transform per compartment, and these are recorded so a reader
#: can reproduce the numbers this script prints.
REPORT_TEMPERATURE_K = 303.15
REPORT_IONIC_STRENGTH_M = 0.25

#: Gas constant in kJ/mol/K, matching eQuilibrator's own value.
R_KJ_PER_MOL_K = 8.31446261815324e-3

#: Which species to extract, and how each ion's element enters its formation reaction.
#: ``elem_row`` is the CODATA row for the element's standard state and ``elem_atoms_per_row``
#: is how many atoms of the element that row's entropy covers, so a diatomic reference state
#: contributes half its tabulated entropy per atom.
WANTED: dict[str, dict[str, object]] = {
    "K": {"ion_row": ("K+", "aq"), "charge": 1, "elem_row": ("K", "cr"), "elem_atoms_per_row": 1},
    "Na": {"ion_row": ("Na+", "aq"), "charge": 1, "elem_row": ("Na", "cr"), "elem_atoms_per_row": 1},
    "Ca": {"ion_row": ("Ca+2", "aq"), "charge": 2, "elem_row": ("Ca", "cr"), "elem_atoms_per_row": 1},
    "Zn": {"ion_row": ("Zn+2", "aq"), "charge": 2, "elem_row": ("Zn", "cr"), "elem_atoms_per_row": 1},
    "Cu": {"ion_row": ("Cu+2", "aq"), "charge": 2, "elem_row": ("Cu", "cr"), "elem_atoms_per_row": 1},
    "Cd": {"ion_row": ("Cd+2", "aq"), "charge": 2, "elem_row": ("Cd", "cr"), "elem_atoms_per_row": 1},
    "Mg": {"ion_row": ("Mg+2", "aq"), "charge": 2, "elem_row": ("Mg", "cr"), "elem_atoms_per_row": 1},
    "Cl": {"ion_row": ("Cl-", "aq"), "charge": -1, "elem_row": ("Cl2", "g"), "elem_atoms_per_row": 2},
}

#: The dihydrogen row, needed by every formation reaction through the proton convention.
H2_ROW = ("H2", "g")

#: The proton row, read only to CHECK that the source uses the convention this derivation
#: assumes: both its formation enthalpy and its entropy must be exactly zero.
PROTON_ROW = ("H+", "aq")

#: Independently published standard Gibbs energies of formation at 298.15 K, kJ/mol, from
#: the NBS tables (Wagman et al. 1982, J. Phys. Chem. Ref. Data 11 Suppl. 2) as reproduced
#: in the CRC Handbook of Chemistry and Physics. A DIFFERENT evaluation from CODATA, used
#: here only to check the derivation; the deviation is reported per ion.
NBS_CHECK_DGF_KJ_PER_MOL = {
    "K": -283.27,
    "Na": -261.905,
    "Ca": -553.58,
    "Zn": -147.06,
    "Cu": 65.49,
    "Cd": -77.612,
    "Mg": -454.8,
    "Cl": -131.228,
}

#: Tolerance on the check. CODATA and NBS are separate evaluations of the same measurements,
#: so they differ by more than either one's stated uncertainty; 2 kJ/mol is the band the
#: eight ions here actually fall in, and a value outside it means the derivation is wrong
#: rather than that the evaluations disagree.
CHECK_TOLERANCE_KJ = 2.0


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


class SourceValue(BaseModel):
    """One tabulated number, with the line it was parsed from."""

    substance: str
    state: str
    quantity: str
    value: float
    uncertainty: float | None
    unit: str
    verbatim_line: str


class IonEntry(BaseModel):
    """One aqueous ion: the source values, the derivation, and the check."""

    element: str
    charge: int
    species: str
    formation_reaction: str
    source_values: list[SourceValue]
    delta_f_h_kj_per_mol: float
    entropy_ion_j_per_mol_k: float
    entropy_element_per_atom_j_per_mol_k: float
    entropy_h2_j_per_mol_k: float
    delta_r_s_j_per_mol_k: float
    delta_f_g_298_15_kj_per_mol: float
    delta_f_g_at_report_temperature_kj_per_mol: float
    report_temperature_k: float
    nbs_check_delta_f_g_kj_per_mol: float | None
    deviation_from_nbs_kj_per_mol: float | None
    #: The transformed value at the reporting conditions, for the reader; the pipeline
    #: applies the transform itself per compartment.
    delta_f_g_prime_at_report_conditions_kj_per_mol: float


class IonFormationTable(BaseModel):
    """The pinned table, its provenance, and the transform it is meant to be used with."""

    built_at: str
    citation: str = CITATION
    source_url: str = SOURCE_URL
    retrieval_method: str = RETRIEVAL_METHOD
    retrieval_command: str = RETRIEVAL_COMMAND
    retrieved_at: str
    pdf_path: str
    pdf_sha256: str
    text_path: str
    text_sha256: str
    extraction_command: str = EXTRACTION_COMMAND
    pdftotext_version: str
    source_temperature_k: float = SOURCE_TEMPERATURE_K
    source_standard_state_pressure_pa: float = 100000.0
    proton_convention_verified: bool
    derivation: str = (
        "M(standard state) + z H+(aq) -> M^z(aq) + (z/2) H2(g); "
        "d_rH = d_fH(ion); d_rS = S(ion) + (z/2) S(H2) - S(element per atom); "
        "d_fG(T) = d_rH - T d_rS, with d_fH(H+,aq) = 0 and S(H+,aq) = 0 by the source's "
        "own convention and d_rS taken temperature independent over the 5 K extrapolation"
    )
    transform: str = (
        "a bare ion has no dissociable protons and no magnesium binding, so the Legendre "
        "transform reduces to the extended Debye-Huckel term: "
        "d_fG'0 = d_fG0 - alpha(T) / (1/sqrt(I) + 1.6) * z^2, with "
        "alpha(T) = a1 T - a2 T^2 + a3 T^3, a1 = 9.20483e-3, a2 = 1.284668e-5, "
        "a3 = 4.95199e-8 kJ/mol per K power"
    )
    transform_matches_equilibrator: bool
    transform_max_abs_deviation_kj_per_mol: float
    check_source: str = (
        "NBS tables (Wagman et al. 1982) as reproduced in the CRC Handbook; a different "
        "evaluation of the same measurements, used only to check the derivation"
    )
    check_tolerance_kj_per_mol: float = CHECK_TOLERANCE_KJ
    max_deviation_from_check_kj_per_mol: float
    report_conditions: dict[str, float]
    n_entries: int
    entries: list[IonEntry] = Field(default_factory=list)
    elements_not_in_source: list[str] = Field(
        default_factory=list,
        description="Elements asked for that the CODATA key values do not tabulate.",
    )
    cancellation_note: str = (
        "In a reaction elementally balanced in the ion's element, the ion's absolute "
        "formation energy enters with coefficients summing to zero and cancels exactly, "
        "as does the ionic-strength term whenever the compartments share an ionic "
        "strength. These values therefore change no balanced reaction's energy; they "
        "matter only for reactions that are not elementally balanced, which in a GEM are "
        "the exchange, sink and biomass pseudo-reactions."
    )


_NUM = r"(-?\d+(?:\.\d+)?)"
_ROW = re.compile(
    rf"^\s*(\S+)\s+(\S+)\s+{_NUM}(?:±{_NUM})?\s+{_NUM}(?:±{_NUM})?"
)


def _parse_rows(text: str) -> dict[tuple[str, str], tuple[float, float | None, float, float | None, str]]:
    """Every ``substance state dfH S`` row of the CODATA table, keyed by (substance, state).

    The layout is fixed-width and the uncertainty is written ``value±sigma`` with no space,
    so one regular expression covers the whole table. Rows whose state field is a phrase
    (``aq,undissoc.``) keep that phrase as the state, which is what makes the undissociated
    species distinguishable from the ion.
    """
    out: dict[tuple[str, str], tuple[float, float | None, float, float | None, str]] = {}
    for line in text.splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        substance, state, h, h_sigma, s, s_sigma = match.groups()
        key = (substance, state)
        if key in out:
            continue
        out[key] = (
            float(h),
            float(h_sigma) if h_sigma else None,
            float(s),
            float(s_sigma) if s_sigma else None,
            line.strip(),
        )
    return out


def debye_hueckel_kj(ionic_strength_m: float, temperature_k: float) -> float:
    """The extended Debye-Huckel coefficient, eQuilibrator's own form and constants."""
    a1, a2, a3 = 9.20483e-3, 1.284668e-5, 4.95199e-8
    alpha = a1 * temperature_k - a2 * temperature_k**2 + a3 * temperature_k**3
    return alpha / (1.0 / ionic_strength_m**0.5 + 1.6)


def transform_kj(charge: int, ionic_strength_m: float, temperature_k: float) -> float:
    """The Legendre transform of a bare ion, kJ/mol, to ADD to the untransformed energy."""
    if ionic_strength_m <= 0:
        return 0.0
    return -debye_hueckel_kj(ionic_strength_m, temperature_k) * charge**2


def main() -> None:
    os.makedirs(MIRROR_DIR, exist_ok=True)
    pdf_path = osp.join(MIRROR_DIR, PDF_NAME)
    text_path = osp.join(MIRROR_DIR, TEXT_NAME)

    retrieved_at = _timestamp()
    if not osp.isfile(pdf_path):
        subprocess.run(
            ["curl", "-sSL", "--fail", "-o", pdf_path, SOURCE_URL], check=True
        )
        print(f"retrieved {pdf_path}")
    else:
        print(f"already mirrored: {pdf_path}")
    subprocess.run(["pdftotext", "-layout", pdf_path, text_path], check=True)
    version = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True
    ).stderr.splitlines()[0].strip()

    with open(text_path) as handle:
        rows = _parse_rows(handle.read())

    proton = rows.get(PROTON_ROW)
    if proton is None:
        raise RuntimeError("the source text has no H+ aq row; the convention is unchecked")
    proton_ok = proton[0] == 0.0 and proton[2] == 0.0
    if not proton_ok:
        raise RuntimeError(
            f"H+ aq is {proton[0]}, {proton[2]} in the source, so it is NOT the zero "
            "reference this derivation assumes"
        )
    h2 = rows.get(H2_ROW)
    if h2 is None:
        raise RuntimeError("the source text has no H2 g row")
    s_h2 = h2[2]

    entries: list[IonEntry] = []
    missing: list[str] = []
    max_deviation = 0.0
    for element, spec in WANTED.items():
        ion_key = tuple(spec["ion_row"])  # type: ignore[arg-type]
        elem_key = tuple(spec["elem_row"])  # type: ignore[arg-type]
        if ion_key not in rows or elem_key not in rows:
            missing.append(element)
            print(f"  {element}: NOT in the CODATA key values ({ion_key}, {elem_key})")
            continue
        charge = int(spec["charge"])  # type: ignore[arg-type]
        per_atom = int(spec["elem_atoms_per_row"])  # type: ignore[arg-type]
        d_f_h, d_f_h_sigma, s_ion, s_ion_sigma, ion_line = rows[ion_key]
        _, _, s_elem_row, s_elem_sigma, elem_line = rows[elem_key]
        s_elem = s_elem_row / per_atom

        d_r_s = s_ion + (charge / 2.0) * s_h2 - s_elem
        d_f_g_298 = d_f_h - SOURCE_TEMPERATURE_K * d_r_s / 1000.0
        d_f_g_t = d_f_h - REPORT_TEMPERATURE_K * d_r_s / 1000.0
        check = NBS_CHECK_DGF_KJ_PER_MOL.get(element)
        deviation = None if check is None else d_f_g_298 - check
        if deviation is not None:
            max_deviation = max(max_deviation, abs(deviation))
            if abs(deviation) > CHECK_TOLERANCE_KJ:
                raise RuntimeError(
                    f"{element}{charge:+d} derived {d_f_g_298:.3f} against the NBS check "
                    f"{check:.3f}, a deviation of {deviation:.3f} kJ/mol which exceeds the "
                    f"{CHECK_TOLERANCE_KJ} kJ/mol tolerance; the derivation is wrong"
                )
        transformed = d_f_g_t + transform_kj(
            charge, REPORT_IONIC_STRENGTH_M, REPORT_TEMPERATURE_K
        )
        entries.append(
            IonEntry(
                element=element,
                charge=charge,
                species=f"{element}{charge:+d}(aq)",
                formation_reaction=(
                    f"{elem_key[0]}({elem_key[1]})"
                    + (f"/{per_atom}" if per_atom != 1 else "")
                    + f" + {charge} H+(aq) -> {element}{charge:+d}(aq)"
                    f" + {charge / 2:g} H2(g)"
                ),
                source_values=[
                    SourceValue(
                        substance=ion_key[0],
                        state=ion_key[1],
                        quantity="delta_f_H(298.15 K)",
                        value=d_f_h,
                        uncertainty=d_f_h_sigma,
                        unit="kJ/mol",
                        verbatim_line=ion_line,
                    ),
                    SourceValue(
                        substance=ion_key[0],
                        state=ion_key[1],
                        quantity="S(298.15 K)",
                        value=s_ion,
                        uncertainty=s_ion_sigma,
                        unit="J/K/mol",
                        verbatim_line=ion_line,
                    ),
                    SourceValue(
                        substance=elem_key[0],
                        state=elem_key[1],
                        quantity="S(298.15 K)",
                        value=s_elem_row,
                        uncertainty=s_elem_sigma,
                        unit="J/K/mol",
                        verbatim_line=elem_line,
                    ),
                    SourceValue(
                        substance=H2_ROW[0],
                        state=H2_ROW[1],
                        quantity="S(298.15 K)",
                        value=s_h2,
                        uncertainty=h2[3],
                        unit="J/K/mol",
                        verbatim_line=h2[4],
                    ),
                ],
                delta_f_h_kj_per_mol=d_f_h,
                entropy_ion_j_per_mol_k=s_ion,
                entropy_element_per_atom_j_per_mol_k=s_elem,
                entropy_h2_j_per_mol_k=s_h2,
                delta_r_s_j_per_mol_k=d_r_s,
                delta_f_g_298_15_kj_per_mol=d_f_g_298,
                delta_f_g_at_report_temperature_kj_per_mol=d_f_g_t,
                report_temperature_k=REPORT_TEMPERATURE_K,
                nbs_check_delta_f_g_kj_per_mol=check,
                deviation_from_nbs_kj_per_mol=deviation,
                delta_f_g_prime_at_report_conditions_kj_per_mol=transformed,
            )
        )
        print(
            f"  {element}{charge:+d}: dfG(298.15) {d_f_g_298:9.3f}"
            f" check {check} dev {None if deviation is None else round(deviation, 3)}"
            f" | dfG'({REPORT_TEMPERATURE_K:g} K, I {REPORT_IONIC_STRENGTH_M}) "
            f"{transformed:9.3f}"
        )

    # The transform this table documents must be eQuilibrator's own, or the ion sits on a
    # different convention from every other metabolite in the table.
    # eQuilibrator's `_legendre_transform` returns the transform DIVIDED by RT, so the
    # comparison has to multiply by ITS gas constant, not the CODATA one. eQuilibrator
    # rounds R to 0.00831 kJ/mol/K, and using the exact value here instead leaves a
    # 0.005 kJ/mol residue that is a unit artifact rather than a disagreement.
    from equilibrator_cache.thermodynamic_constants import (
        _legendre_transform,
        default_R_in_kJ_per_mol_per_K,
    )

    worst = 0.0
    for charge in (-2, -1, 1, 2, 3):
        for ionic in (0.1, 0.25, 0.5):
            for temperature in (298.15, 303.15, 310.15):
                rt = default_R_in_kJ_per_mol_per_K * temperature
                reference = rt * _legendre_transform(
                    pH=7.2,
                    pMg=3.0,
                    ionic_strength_M=ionic,
                    T_in_K=temperature,
                    charge=charge,
                    num_protons=0,
                    num_magnesiums=0,
                )
                worst = max(worst, abs(reference - transform_kj(charge, ionic, temperature)))
    if worst > 1e-9:
        raise RuntimeError(
            f"the recorded transform differs from eQuilibrator's by {worst} kJ/mol"
        )

    table = IonFormationTable(
        built_at=_timestamp(),
        retrieved_at=retrieved_at,
        pdf_path=pdf_path,
        pdf_sha256=_sha256(pdf_path),
        text_path=text_path,
        text_sha256=_sha256(text_path),
        pdftotext_version=version,
        proton_convention_verified=proton_ok,
        transform_matches_equilibrator=worst <= 1e-9,
        transform_max_abs_deviation_kj_per_mol=worst,
        max_deviation_from_check_kj_per_mol=max_deviation,
        report_conditions={
            "temperature_k": REPORT_TEMPERATURE_K,
            "ionic_strength_m": REPORT_IONIC_STRENGTH_M,
        },
        n_entries=len(entries),
        entries=entries,
        elements_not_in_source=missing,
    )
    for path in (osp.join(MIRROR_DIR, TABLE_NAME), osp.join(RESULTS, TABLE_NAME)):
        with open(path, "w") as handle:
            handle.write(table.model_dump_json(indent=1))
        print(f"wrote {path}")
    print(
        f"  {table.n_entries} ions, worst deviation from the NBS check "
        f"{table.max_deviation_from_check_kj_per_mol:.3f} kJ/mol, transform matches "
        f"eQuilibrator to {table.transform_max_abs_deviation_kj_per_mol:.2e} kJ/mol"
    )


if __name__ == "__main__":
    main()

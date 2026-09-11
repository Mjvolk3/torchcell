# tests/torchcell/metabolism/test_kinetics_pipeline.py

"""The portable kinetics pipeline, on a synthetic GEM with synthetic mirrors.

Synthetic on purpose. The point of the pipeline is that it carries no organism-specific
knowledge, so the test that proves it is one where every input is invented: a four-reaction
model, a five-protein proteome, a two-row structure table, a one-row kinetics mirror and a
one-column predictor table. If any yeast assumption were left in the code, none of this
would resolve.

Each blocker class gets its own reaction, because a coverage number that does not partition
the units is the failure mode this pipeline exists to prevent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cobra
import pandas as pd
import pytest

from torchcell.metabolism.enzyme_kinetics import KineticRetrieval, _sha256
from torchcell.metabolism.kinetics_pipeline import (
    Blocker,
    KineticsPipelineConfig,
    MeasuredKey,
    PredictorTable,
    StructureRoute,
    Tier,
    neutral_formula,
    resolve_kinetics,
    structure_matches_metabolite,
)

GLUCOSE = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
PYRUVATE = "CC(=O)C(=O)O"
LACTATE = "CC(O)C(=O)O"


def test_neutral_formula_moves_an_anion_to_its_acid() -> None:
    """A GEM's charged species and a database's neutral acid must compare equal."""
    assert neutral_formula("C6H10O12P2", -4.0) == neutral_formula("C6H14O12P2", 0.0)


def test_a_placeholder_never_matches_a_real_composition() -> None:
    """An R-group species has no structure, and no SMILES may be accepted for it."""
    assert not structure_matches_metabolite("CCC(=O)S", "C4H7O2SR", 0.0)
    assert structure_matches_metabolite(PYRUVATE, "C3H3O3", -1.0)


def build_model() -> cobra.Model:
    """Four reactions, one per outcome the pipeline has to distinguish."""
    model = cobra.Model("synthetic")
    glucose = cobra.Metabolite(
        "m_glc", formula="C6H12O6", charge=0, name="D-glucose", compartment="c"
    )
    pyruvate = cobra.Metabolite(
        "m_pyr", formula="C3H3O3", charge=-1, name="pyruvate", compartment="c"
    )
    lactate = cobra.Metabolite(
        "m_lac", formula="C3H5O3", charge=-1, name="lactate", compartment="c"
    )
    acyl = cobra.Metabolite(
        "m_acp",
        formula="C4H7O2SR",
        charge=0,
        name="3-hydroxybutanoyl-ACP",
        compartment="c",
    )
    crotonyl = cobra.Metabolite(
        "m_cro", formula="C4H5O2R", charge=0, name="crotonoyl-ACP", compartment="c"
    )
    glucose.annotation = {"chebi": "CHEBI:4167"}
    pyruvate.annotation = {"kegg.compound": "C00022"}
    lactate.annotation = {"metanetx.chemical": "MNXOLD"}

    glycolysis = cobra.Reaction("r_glyc")
    glycolysis.add_metabolites({glucose: -1.0, pyruvate: 2.0})
    glycolysis.gene_reaction_rule = "(g1 and g2) or g3"
    glycolysis.annotation = {"ec-code": "1.1.1.1"}

    reduction = cobra.Reaction("r_ldh")
    reduction.add_metabolites({pyruvate: -1.0, lactate: 1.0})
    reduction.gene_reaction_rule = "g4"
    reduction.annotation = {"ec-code": "1.1.1.27"}

    dehydratase = cobra.Reaction("r_acp")
    dehydratase.add_metabolites({acyl: -1.0, crotonyl: 1.0})
    dehydratase.gene_reaction_rule = "g5"

    supply = cobra.Reaction("r_supply")
    supply.add_metabolites({glucose: 1.0})
    supply.gene_reaction_rule = "g6"

    model.add_reactions([glycolysis, reduction, dehydratase, supply])
    return model


def write_inputs(tmp_path: Path) -> KineticsPipelineConfig:
    """Every mirror the pipeline reads, written small enough to read in the test."""
    model = build_model()
    sbml = str(tmp_path / "model.xml")
    cobra.io.write_sbml_model(model, sbml)

    # g2 is deliberately absent: its unit is a complex, so the complex rule must refuse it
    # while the isozyme g3 on the same reaction still resolves.
    proteome = {
        "g1": "MKAILVVLLYTFATANADTLC",
        "g3": "MSTAGKVIKCKAAVLWEEKKP",
        "g4": "MRALIVLAHSERTSFNYAMKE",
        "g5": "MSEAPKQLTLAQFRRAVDEML",
        "g6": "MPQRSTVWYACDEFGHIKLMN",
    }
    proteome_path = str(tmp_path / "proteome.json")
    with open(proteome_path, "w") as handle:
        json.dump(proteome, handle)
    accessions_path = str(tmp_path / "accessions.json")
    with open(accessions_path, "w") as handle:
        json.dump({"g1": "P00001", "g3": "P00003", "g4": "P00004"}, handle)

    name_table = str(tmp_path / "names.tsv")
    with open(name_table, "w") as handle:
        handle.write(f"D-glucose\t{GLUCOSE}\n")

    # The structure table is keyed by a CURRENT id and the GEM annotates an OLD one, which
    # is the situation the cross-reference route exists for.
    chem_prop = str(tmp_path / "chem_prop.tsv")
    with open(chem_prop, "w") as handle:
        handle.write(
            "#ID\tname\treference\tformula\tcharge\tmass\tInChI\tInChIKey\tSMILES\n"
        )
        handle.write(f"MNXNEW\tpyruvate\tx\tC3H4O3\t0\t88\t\t\t{PYRUVATE}\n")
        handle.write(f"MNXLAC\tlactate\tx\tC3H6O3\t0\t90\t\t\t{LACTATE}\n")
    chem_xref = str(tmp_path / "chem_xref.tsv")
    with open(chem_xref, "w") as handle:
        handle.write("#source\tID\tdescription\n")
        handle.write("kegg.compound:C00022\tMNXNEW\tpyruvate\n")
        handle.write("MNXOLD\tMNXLAC\tlactate\n")

    mirror_dir = tmp_path / "kinetics_mirror"
    mirror_dir.mkdir()
    rows = [
        {
            "ec": "1.1.1.1",
            "enzymetype": "wildtype",
            "organism": "Synthetica minima",
            "uniprot": "P00003",
            "substrate": "D-glucose",
            "smiles": GLUCOSE,
            "temperature": 30.0,
            "ph": 7.0,
            "kcat_value": 42.0,
            "kcat_unit": "1/s",
            "kcat_pubmedid": 1.0,
            "km_value": 0.5,
            "km_unit": "mM",
            "km_pubmedid": 1.0,
        }
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    with open(mirror_dir / "oed_records.json", "wb") as handle:
        handle.write(payload)
    manifest = KineticRetrieval(
        source_url="synthetic",
        retrieval_command="synthetic",
        sha256=_sha256(payload),
        retrieved_at=datetime.now(UTC),
        n_records=len(rows),
        organism="Synthetica minima",
    )
    with open(mirror_dir / "manifest.json", "w") as handle:
        handle.write(manifest.model_dump_json(indent=2))

    # Unit ids follow the GPR parse order: 0 = (g1 and g2), 1 = g3, 2 = g4, 3 = g5, 4 = g6.
    predictions = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "gene_id": ["g3", "g4"],
            "k_cat": [7.0, 3.0],
            "K_M": [0.2, 0.4],
        }
    )
    predictor_path = str(tmp_path / "predictor.parquet")
    predictions.to_parquet(predictor_path, index=False)

    return KineticsPipelineConfig(
        sbml_path=sbml,
        proteome_path=proteome_path,
        accessions_path=accessions_path,
        name_table_path=name_table,
        chem_prop_path=chem_prop,
        chem_xref_path=chem_xref,
        measured_mirror_dir=str(mirror_dir),
        measured_key=MeasuredKey.UNIPROT,
        organism="Synthetica minima",
        predictor_tables={
            "k_cat": [PredictorTable(name="toy", path=predictor_path)],
            "K_M": [PredictorTable(name="toy", path=predictor_path)],
        },
        selected_predictors={"k_cat": ["toy"], "K_M": ["toy"]},
    )


def test_pipeline_partitions_every_unit_into_a_tier_or_a_blocker(
    tmp_path: Path,
) -> None:
    """Tiers plus blockers must account for every catalytic unit, with no unit twice."""
    config = write_inputs(tmp_path)
    result = resolve_kinetics(config)

    assert result.n_catalytic_units == 5
    block = result.parameters["k_cat"]
    assert len(block.values) == result.n_catalytic_units
    assert sum(block.coarse_tier_counts().values()) == result.n_catalytic_units

    by_unit = {v.unit_id: v for v in block.values}
    genes = {v.unit_id: tuple(v.gene_ids) for v in block.values}
    complex_unit = next(u for u, g in genes.items() if set(g) == {"g1", "g2"})
    isozyme_unit = next(u for u, g in genes.items() if g == ("g3",))
    ldh_unit = next(u for u, g in genes.items() if g == ("g4",))
    acp_unit = next(u for u, g in genes.items() if g == ("g5",))
    supply_unit = next(u for u, g in genes.items() if g == ("g6",))

    # The accession-keyed measurement wins over the prediction on the same unit.
    assert by_unit[isozyme_unit].tier is Tier.MEASURED_PROTEIN
    assert by_unit[isozyme_unit].value == pytest.approx(42.0)
    isozyme_measured = by_unit[isozyme_unit].measured
    assert isozyme_measured is not None
    assert isozyme_measured.organism == "Synthetica minima"

    # A predicted unit takes the predictor's value and names the predictor.
    assert by_unit[ldh_unit].tier is Tier.PREDICTED
    assert by_unit[ldh_unit].value == pytest.approx(3.0)
    assert by_unit[ldh_unit].predictors == ["toy"]

    # One blocker per remaining unit, and each a different one.
    assert by_unit[complex_unit].blocker is Blocker.COMPLEX_RULE_UNHANDLED
    assert by_unit[acp_unit].blocker is Blocker.SUBSTRATE_STRUCTURE_MISSING
    assert by_unit[supply_unit].blocker is Blocker.REACTION_HAS_NO_SUBSTRATE
    for unit in (complex_unit, acp_unit, supply_unit):
        assert by_unit[unit].tier is Tier.DEFAULT
        assert by_unit[unit].value == pytest.approx(42.0), (
            "the default is the median of what the database supplied, not a constant"
        )


def test_structures_come_from_the_route_that_should_have_supplied_them(
    tmp_path: Path,
) -> None:
    """The name table, the cross-reference and the refusal each have to fire once."""
    config = write_inputs(tmp_path)
    result = resolve_kinetics(config)
    assert result.structures_by_route == {
        str(StructureRoute.NAME_TABLE): 1,
        str(StructureRoute.ANNOTATION_XREF): 1,
    }, "glucose by name, pyruvate through the xref, the ACP placeholder refused"
    assert result.n_substrates_resolved == 2


def test_an_ec_key_reaches_a_unit_whose_protein_was_never_assayed(
    tmp_path: Path,
) -> None:
    """The portable measured tier: same chemistry, no accession, tagged as a homolog."""
    config = write_inputs(tmp_path)
    config.measured_key = MeasuredKey.EC_SUBSTRATE_ANY_ORGANISM
    config.organism = None
    result = resolve_kinetics(config)
    block = result.parameters["k_cat"]
    by_unit = {v.unit_id: v for v in block.values}
    genes = {v.unit_id: tuple(v.gene_ids) for v in block.values}
    complex_unit = next(u for u, g in genes.items() if set(g) == {"g1", "g2"})
    isozyme_unit = next(u for u, g in genes.items() if g == ("g3",))
    assert by_unit[isozyme_unit].tier is Tier.MEASURED_EC_HOMOLOG
    # The complex is measured too, and that is correct rather than a leak: an EC key asks
    # about the reaction's chemistry, so it does not need a sequence. The missing subunit
    # blocks the PREDICTED tier, which is what the blocker accounting is about.
    assert by_unit[complex_unit].tier is Tier.MEASURED_EC_HOMOLOG
    complex_measured = by_unit[complex_unit].measured
    assert complex_measured is not None
    assert complex_measured.key is MeasuredKey.EC_SUBSTRATE_ANY_ORGANISM


def test_a_run_with_no_mirrors_still_accounts_for_every_unit(tmp_path: Path) -> None:
    """With no measured tier and no predictors, every unit is a default with a reason."""
    config = write_inputs(tmp_path)
    config.measured_mirror_dir = None
    config.predictor_tables = {}
    result = resolve_kinetics(config)
    block = result.parameters["k_cat"]
    assert block.coarse_tier_counts()["default"] == result.n_catalytic_units
    assert all(v.blocker is not None for v in block.values)
    assert block.default_value == pytest.approx(config.fallback_defaults["k_cat"])

# experiments/019-simb-multimodal/scripts/decoder_arms_table.py
# [[experiments.019-simb-multimodal.scripts.decoder_arms_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/decoder_arms_table
"""Tabulate the wave 1 to 3 decoder and perturbation-family arms of the v8 expression round.

The expression document needs one table answering "was a head or decoder family
comparison ever run": the GEARS-style cross-gene readout, the rank-32 bilinear pair term,
Perceiver post-perturbation mixing, the response basis, context concatenation, graph
propagation, and the deeper perturbation heads were all trained in waves 1 to 3 of project
`torchcell_019_expr_v8` (`score_decoder_arms.py`, results/decoder_arms_torchcell_019_expr_v8.csv).
Every one of those runs stopped at or below 276 epochs, against a validation curve that has
not turned by 9,900, so the table is a record of what was tried and at what budget, not a
comparison. It says so in its caption.

Each arm's mechanism is read from its config override in `gh_expr_008_arm.sh`, the file
the launcher actually ran, rather than retyped; the arm-name to override map is parsed
from that script at build time, so a renamed arm fails loudly here.

Writes notes-tex/019-simb-multimodal-expression/tables/decoder_arms_v8.tex and
results/decoder_arms_v8_table.csv. Run from the repository root:

    python experiments/019-simb-multimodal/scripts/decoder_arms_table.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "019-simb-multimodal"
ARMS_CSV = EXP / "results" / "decoder_arms_torchcell_019_expr_v8.csv"
ARM_SCRIPT = EXP / "scripts" / "gh_expr_008_arm.sh"
TABLE_OUT = (
    REPO_ROOT / "notes-tex" / "019-simb-multimodal-expression" / "tables" / "decoder_arms_v8.tex"
)
CSV_OUT = EXP / "results" / "decoder_arms_v8_table.csv"

# The waves whose arms vary the decoder or the perturbation operator. Waves 4a and 4b are
# the null-sink stability pairs and the learning-rate arms are optimizer settings; both are
# excluded so the table answers the decoder-family question only.
WAVES = ("wave1", "wave2", "wave3")
EXCLUDE_PREFIXES = ("L0_", "L1_", "L2_", "L3_", "S1_", "S2_", "S3_")

# What each override family means, keyed by the first override token. This is the only
# hand-written text in the table and it is a gloss on the override, which is printed beside
# it.
GLOSS = {
    "": "additive reference, $g(h_i + c_b)$",
    "multitask.free_gene_dim": "free per-gene lookup of 16 dims",
    "model.perturbation_propagation.hops=0": "self-indicator only (hop 0)",
    "model.perturbation_propagation.hops=2": "random-walk propagation, 2 hops",
    "model.perturbation_propagation.hops=2 sparse": "propagation, 2 hops, sparse",
    "cell_dataset.node_embeddings": "ProtT5 gene embeddings",
    "multitask.concat_context": "head sees $[h_{\\mathrm{pert}}; h_i; c_b]$",
    "multitask.bilinear_rank": "rank-32 bilinear pair term",
    "model.post_perturbation_mixing.enabled": "Perceiver mixing after the perturbation",
    "model.attention_mask.enabled": "hard graph mask replaces the KL",
    "model.perturbation_head.num_layers": "deeper perturbation head",
    "model.cross_gene.enabled": "GEARS-style cross-gene readout, rank 64",
    "multitask.response_basis_rank": "rank-32 response basis (factored readout)",
    "model.perturbation_head.null_sink": "null-sink gated attention",
}


class ArmRow(BaseModel):
    """One arm of the v8 decoder round: what varied, how many seeds, how far it ran."""

    arm: str
    wave: str
    override: str
    mechanism: str
    n_seeds: int
    epochs_min: int
    epochs_max: int
    smoothed_mean: float
    smoothed_best: float


def parse_overrides(script: Path) -> dict[str, str]:
    """Map arm name to its OVERRIDES=(...) string, spanning continuation lines."""
    text = script.read_text()
    out: dict[str, str] = {}
    for m in re.finditer(r"^\s*([A-Za-z0-9_]+)\)\s+OVERRIDES=\((.*?)\)", text, re.M | re.S):
        arm, body = m.group(1), " ".join(m.group(2).split())
        out.setdefault(arm, body)
    return out


def gloss(arm: str, override: str) -> str:
    if override == "":
        return GLOSS[""]
    if "hops=2" in override and "sparse" in arm:
        return GLOSS["model.perturbation_propagation.hops=2 sparse"]
    if "hops=2" in override:
        return GLOSS["model.perturbation_propagation.hops=2"]
    if "hops=0" in override:
        return GLOSS["model.perturbation_propagation.hops=0"]
    for key, text in GLOSS.items():
        if key and key in override:
            return text
    raise KeyError(f"no gloss for {arm}: {override}")


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("$PROP", "\\$PROP").replace("%", "\\%")


def tex_override(s: str) -> str:
    """An override string set in typewriter that can break at its own punctuation.

    A Hydra override is one unbreakable token per key, and the graph list alone is 80
    characters, so without break points the table runs 67 mm past the text block.
    """
    out = tex_escape(s)
    for ch in (".", "=", ",", "["):
        out = out.replace(ch, ch + "\\allowbreak{}")
    return out


def build() -> list[ArmRow]:
    t = pd.read_csv(ARMS_CSV)
    t = t[t.wave.isin(WAVES) & ~t.arm.str.startswith(EXCLUDE_PREFIXES)]
    overrides = parse_overrides(ARM_SCRIPT)
    rows: list[ArmRow] = []
    for (arm, wave), g in t.groupby(["arm", "wave"], sort=False):
        ov = overrides[arm]
        rows.append(ArmRow(
            arm=arm, wave=wave, override=ov, mechanism=gloss(arm, ov),
            n_seeds=int(g.seed.nunique()), epochs_min=int(g.epochs.min()),
            epochs_max=int(g.epochs.max()), smoothed_mean=float(g.smoothed.mean()),
            smoothed_best=float(g.smoothed.max()),
        ))
    rows.sort(key=lambda r: (WAVES.index(r.wave), r.arm))
    return rows


def write_table(rows: list[ArmRow]) -> None:
    n_runs = sum(r.n_seeds for r in rows)
    lo = min(r.epochs_min for r in rows)
    hi = max(r.epochs_max for r in rows)
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/decoder_arms_table.py",
        "%% from results/decoder_arms_torchcell_019_expr_v8.csv (score_decoder_arms.py) and",
        "%% the arm overrides in scripts/gh_expr_008_arm.sh. Do not edit by hand.",
        "%% SOURCE: the two files named above.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Decoder and perturbation-family arms of the v8 round]{The decoder and "
        "perturbation-family arms of waves 1 to 3 in project \\file{torchcell_019_expr_v8}, "
        f"{n_runs} runs over {len(rows)} arm-wave cells, scored by the maximum of a centered "
        "5-epoch rolling mean of the validation Pearson (\\file{score_decoder_arms.py}). "
        f"Every run stopped between epochs {lo} and {hi}, inside the early dip of a curve "
        "that is still rising at 9{,}900 (\\cref{sec:findings-lossmin}), so the scores "
        "measure early-training transients and the column is not an arm comparison. The "
        "GEARS-style cross-gene readout, the bilinear pair term, Perceiver mixing and the "
        "response basis are all here, each at one seed. The mechanism column is a gloss on "
        "the override, which is the launcher's own text.}",
        "  \\label{tab:decoder-arms-v8}",
        "  \\begin{tabular}{@{}ll>{\\raggedright\\arraybackslash}p{0.30\\textwidth}rrr@{}}",
        "    \\hline",
        "    Arm & Wave & Mechanism (override) & Seeds & Epochs & Smoothed max, mean / best \\\\",
        "    \\hline",
    ]
    for r in rows:
        ep = f"{r.epochs_min}" if r.epochs_min == r.epochs_max else f"{r.epochs_min}--{r.epochs_max}"
        ov = tex_override(r.override) if r.override else "none"
        lines.append(
            f"    \\texttt{{{tex_escape(r.arm)}}} & {r.wave.replace('wave', '')} & "
            f"{r.mechanism} (\\texttt{{\\scriptsize {ov}}}) & {r.n_seeds} & {ep} & "
            f"${r.smoothed_mean:.3f}$ / ${r.smoothed_best:.3f}$ \\\\"
        )
    lines += ["    \\hline", "  \\end{tabular}", "\\end{table}", ""]
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines))


def main() -> None:
    rows = build()
    pd.DataFrame([r.model_dump() for r in rows]).to_csv(CSV_OUT, index=False)
    write_table(rows)
    print(f"{len(rows)} arm-wave cells, {sum(r.n_seeds for r in rows)} runs")
    for r in rows:
        print(f"  {r.arm:18s} {r.wave:6s} seeds {r.n_seeds} epochs {r.epochs_min}-{r.epochs_max} "
              f"smoothed {r.smoothed_mean:.3f}/{r.smoothed_best:.3f}  {r.mechanism}")
    print(f"wrote {TABLE_OUT}\nwrote {CSV_OUT}")


if __name__ == "__main__":
    main()

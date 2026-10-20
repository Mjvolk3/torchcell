# experiments/040-inhibitor-synergy-wetlab/scripts/query_mixture.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.query_mixture]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/query_mixture
r"""Pose a counterfactual to a trained mixture model: a genotype, a medium, or a dose.

``train_mixture.py`` writes one checkpoint per (config, fold, seed) holding the weights,
the compound feature scaling, the dose scaling and the condition bookkeeping. This script
loads one or several of them and answers four kinds of question the score tables cannot,
because each asks for a prediction at a point no experiment visited.

``host``      the host's fitness in a medium: any set of compounds at any doses, named
              from the compound table or given as SMILES. This is the single-agent curve,
              the combination plate and anything else the wet lab could run next.
``grid``      a dose grid of two compounds, the predicted fitness surface, its alive and
              dead regions under the grid's own growth threshold, and the monotone closure
              of that region (``mixture_rules.monotone_alive``: growth never returns as
              dose rises on either axis). Where the pair and doses match one of the three
              2021 isoboles the observed boundary is scored against the predicted one by
              the same Jaccard overlap and growth-edge error the model-free rules are
              scored by, so the model's front is read on the rules' own axis. The closure
              is also the jaggedness readout: the cells it drops are the predicted
              surface's growth islands, which a rule-based front cannot have.
``profile``   the gene-level response under a medium, for every gene of a source or for
              ANY gene of the cell graph. A gene outside the source panel has no fitted
              gene bias (the bias is per source and per node and initialized to zero), so
              its prediction comes from the cell graph and the medium alone. That is the
              inductive claim the in-house deletion assay would test, and the ranked table
              is the list such an assay would screen.
``genotype``  the same, for a named genotype rather than a source's single deletions: any
              set of deleted genes, which the readout sums over, so a double or triple
              deletion under a mixture is a legal question even though no training record
              is one.

EVERY PREDICTION IS A MODEL OUTPUT AND NOTHING ELSE. The tables carry the checkpoint, the
fold and the seed in every row; a row whose compound is a SMILES the model never saw and a
row whose gene is outside the fitted panel are both extrapolations, and the ``in_training``
columns say which. Averaging over the seeds and folds of one arm is the ensemble the score
tables use, and ``--mean`` does it.

    python query_mixture.py --list
    python query_mixture.py host --checkpoint <arm>_fold0_fs0_seed0.pt \
        --medium "acetic acid=2.0,furfural=1.5" --medium "acetic acid=2.0"
    python query_mixture.py grid --arm r3_vanhop_e150 --mean \
        --pair "furfural,acetic acid" --run ex26
    python query_mixture.py profile --arm r3_van_e150_gene --mean \
        --medium "furfural=1.5" --genes-outside-panel --top 50
    python query_mixture.py genotype --arm r3_vanhop_e150 \
        --genotype YGL013C,YBL005W --medium "acetic acid=2.0,5-HMF=1.0"

Writes ``results/query/<name>.csv`` and prints the head of every table.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

sys.path.insert(0, osp.dirname(__file__))
from mixture_data import (  # noqa: E402
    ABBREVIATION,
    IDENTITY_GAP_SMILES,
    build_cell_graph,
    fcfp4_count,
    identity_smiles,
    wetlab_inhibitors,
)
from mixture_rules import ISOBOLES, isobole_grid, monotone_alive  # noqa: E402
from train_mixture import (  # noqa: E402
    CHECKPOINTS,
    MODEL_YAML,
    RESULTS,
    WELLS_CSV,
    MixtureConfig,
    MixtureModel,
    build_encoder,
    group_media,
)

OUT_DIR = osp.join(RESULTS, "query")


class Model:
    """One loaded checkpoint: the weights, the scales, and what it was fitted on."""

    def __init__(self, path: str, cell_graph, node_ids: list[str], device) -> None:
        """Load ``path`` and rebuild its model on ``cell_graph``."""
        state = torch.load(path, map_location=device, weights_only=False)
        self.path = path
        self.name = osp.splitext(osp.basename(path))[0]
        self.cfg = MixtureConfig(**state["config"])
        self.fold = state["fold"]
        self.seed = int(state["seed"])
        self.selected_step = int(state["selected_step"])
        self.device = device
        self.cell_graph = cell_graph
        self.node_ids = node_ids
        self.position = {gene: i for i, gene in enumerate(node_ids)}
        std = state["standardization"]
        self.x = std["x"].to(device)
        self.x_keep = std.get("x_keep")
        self.x_mean = std.get("x_mean")
        self.x_sd = std.get("x_sd")
        self.dose_center = float(std["dose_center"])
        self.dose_scale = float(std["dose_scale"])
        self.y_mean = std["y_mean"]
        self.y_sd = std["y_sd"]
        self.inchikeys = list(state["compounds"]["inchikeys"])
        self.compound_names = list(state["compounds"]["names"])
        self.row_of_name = {
            name.lower(): i for i, name in enumerate(self.compound_names)
        }
        self.sources = state["sources"]
        self.source_of = {s["name"]: s for s in self.sources}
        # the compound rows and the genes the model was FITTED on, so a query can say
        # whether it is asking inside or outside the training set
        fitted_rows: set[int] = set()
        van = self.sources[0]
        fitted_compounds = (
            list(range(len(van["compounds"])))
            if self.cfg.split == "gene"
            else sorted(set(self.fold["train"]) | set(self.fold["val"]))
            if self.cfg.fit_on == "pool"
            else list(self.fold["train"])
        )
        for j in fitted_compounds:
            fitted_rows.add(int(van["compound_row"][j]))
        for name, conditions in state["aux_conditions"].items():
            source = self.source_of[name]
            for j in conditions:
                fitted_rows.add(int(source["compound_row"][j]))
        self.fitted_rows = fitted_rows
        self.fitted_genes = (
            set(van["genes"][i] for i in self.fold["train"])
            if self.cfg.split == "gene"
            else set(van["genes"])
        )
        self.extra_genes = tuple(van["extra_genes"])
        self.model = MixtureModel(
            self.cfg,
            n_nodes=len(node_ids),
            n_sources=len(self.sources),
            feature_dim=self.x.shape[1],
            encoder=build_encoder(self.cfg, cell_graph),
            source_attributes=state["source_attributes"],
            n_attribute_values=[len(v) for v in state["attribute_values"].values()],
        ).to(device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self._x_extra: list[torch.Tensor] = []

    # ---- building a question ---------------------------------------------- #
    def compound_row(self, key: str) -> tuple[int, bool]:
        """(feature row, whether the model was fitted on it) for a name or a SMILES.

        A name is matched case-insensitively against the compound table, including the
        six 2021 inhibitors and their abbreviations. Anything else is read as SMILES and
        featurized by the same fingerprint, then put on this model's feature scale, which
        needs a checkpoint written after the scale was recorded.
        """
        alias = {v.lower(): k.lower() for k, v in ABBREVIATION.items()}
        lookup = alias.get(key.lower(), key.lower())
        if lookup in self.row_of_name:
            row = self.row_of_name[lookup]
            return row, int(row) in self.fitted_rows
        assert self.x_mean is not None, (
            f"{self.name}: this checkpoint predates the feature scale, so a compound "
            f"outside the table ({key}) cannot be put on its scale; rerun the arm"
        )
        raw = fcfp4_count(key).astype(np.float64)
        scaled = (raw[self.x_keep] - self.x_mean[self.x_keep]) / self.x_sd[self.x_keep]
        self._x_extra.append(
            torch.tensor(scaled, dtype=torch.float32, device=self.device)
        )
        return self.x.shape[0] + len(self._x_extra) - 1, False

    def features(self) -> torch.Tensor:
        """The feature matrix, with any SMILES compounds this query added appended."""
        if not self._x_extra:
            return self.x
        return torch.cat([self.x, torch.stack(self._x_extra)], dim=0)

    def strain(self, genes: list[str]) -> torch.Tensor:
        """[1, order] cell-graph indices of a genotype, with the source's host deletions."""
        missing = [g for g in genes if g not in self.position]
        assert not missing, f"outside the cell graph: {missing}"
        index = [self.position[g] for g in genes] + [
            self.position[g] for g in self.extra_genes
        ]
        return torch.tensor([index], dtype=torch.long, device=self.device)

    # ---- asking it --------------------------------------------------------- #
    @torch.no_grad()
    def host(self, media: list[list[tuple[int, float]]]) -> NDArray[np.float64]:
        """Predicted host fitness for each medium, given as (feature row, log10 molar)."""
        strain = self.strain([self.sources[0]["genes"][0]])
        out = np.full(len(media), np.nan)
        groups = group_media(
            [[row for row, _ in m] for m in media],
            [[dose for _, dose in m] for m in media],
            self.features(),
            self.dose_center,
            self.dose_scale,
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            h, _, _ = self.model.cell(strain, self.cell_graph)
            for medium in groups:
                e = self.model.tokens(medium)
                _, pooled, token = self.model.medium_state(h, e, None)
                out[np.array(medium.rows)] = (
                    self.model.host_logits(pooled, token, e).float().cpu().numpy()
                )
        return out

    @torch.no_grad()
    def gene_response(
        self,
        genotypes: list[list[str]],
        medium: list[tuple[int, float]],
        source: str,
        batch: int = 256,
    ) -> NDArray[np.float64]:
        """Predicted response of each genotype under one medium, in the source's units.

        The genotype enters as the cell-graph rows the readout sums over, so a set of any
        size is legal. The source names whose units and whose gene bias are used.
        """
        index = self.source_of[source]["index"]
        groups = group_media(
            [[row for row, _ in medium]],
            [[dose for _, dose in medium]],
            self.features(),
            self.dose_center,
            self.dose_scale,
        )
        assert len(groups) == 1
        out = np.full(len(genotypes), np.nan)
        # the readout sums over the strain's deleted rows, so a batch has to be of one
        # genotype ORDER; genotypes of different order go in different batches, as media
        # of different compound count do
        by_order: dict[int, list[int]] = {}
        for i, genes in enumerate(genotypes):
            by_order.setdefault(len(genes), []).append(i)
        for order in sorted(by_order):
            positions = by_order[order]
            for start in range(0, len(positions), batch):
                chunk = positions[start : start + batch]
                strain = torch.cat([self.strain(genotypes[i]) for i in chunk])
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    h, h_del, _ = self.model.cell(strain, self.cell_graph)
                    e = self.model.tokens(groups[0])
                    env, pooled, token = self.model.medium_state(h, e, strain)
                    assert env is not None
                    pred = self.model.gene_logits(
                        env, h_del, pooled, token, strain, index, e
                    )
                out[np.array(chunk)] = pred.float().cpu().numpy()[:, 0]
        return out * self.y_sd[source] + self.y_mean[source]


def load_models(args, device) -> list[Model]:
    """The checkpoints the arguments name, as loaded models sharing one cell graph."""
    import yaml

    with open(MODEL_YAML) as f:
        graphs = list(yaml.safe_load(f)["cell_dataset"]["graphs"])
    cell_graph = build_cell_graph(graphs).to(device)
    node_ids = list(cell_graph["gene"].node_ids)
    if args.checkpoint:
        paths = [
            p if osp.isabs(p) else next(iter(glob(osp.join(CHECKPOINTS, "*", p))), p)
            for p in args.checkpoint
        ]
    else:
        assert args.arm, "name --arm or --checkpoint"
        paths = sorted(glob(osp.join(CHECKPOINTS, args.arm, "*.pt"))) or sorted(
            glob(osp.join(CHECKPOINTS, "*", f"mix_{args.arm}_*.pt"))
        )
    assert paths, f"no checkpoint matched (looked under {CHECKPOINTS})"
    for path in paths:
        assert osp.exists(path), f"no checkpoint at {path}"
    return [Model(path, cell_graph, node_ids, device) for path in paths]


def parse_medium(model: Model, text: str) -> tuple[list[tuple[int, float]], list[str]]:
    """``"acetic acid=2.0,furfural=1.5"`` as (feature row, log10 molar) pairs.

    A dose is in mM, the unit the wet-lab table carries, and is converted to log10 molar
    as the trainer's host records are. An empty string is the compound-free medium.
    """
    pairs, labels = [], []
    for part in (p.strip() for p in text.split(",") if p.strip()):
        assert "=" in part, f"write a medium as name=mM, not {part!r}"
        key, value = part.rsplit("=", 1)
        row, fitted = model.compound_row(key.strip())
        pairs.append((row, float(np.log10(float(value) * 1e-3))))
        labels.append(f"{key.strip()}={value}mM{'' if fitted else ' (unfitted)'}")
    return pairs, labels


def molar_mass(name: str) -> float:
    """Grams per mole of a wet-lab inhibitor, from its curated structure.

    The 2021 doses are g/L and the model reads log10 molar, so an isobole grid cannot be
    put to the model without the molar mass. The structure is the one the trainer predicts
    from: the private loader's for four inhibitors, ``IDENTITY_GAP_SMILES`` for formic and
    lactic acid, whose identity the loader leaves open.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    alias = {v.lower(): k for k, v in ABBREVIATION.items()}
    key = alias.get(name.lower(), name)
    inchikey_of = wetlab_inhibitors()
    assert key in inchikey_of, f"not a 2021 inhibitor: {name}"
    smiles = IDENTITY_GAP_SMILES.get(key) or identity_smiles().get(inchikey_of[key])
    assert smiles, f"no curated structure for {name}"
    return float(Descriptors.MolWt(Chem.MolFromSmiles(smiles)))


def write(frame: pd.DataFrame, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = osp.join(OUT_DIR, f"{name}.csv")
    frame.to_csv(path, index=False)
    print(frame.head(25).to_string(index=False), flush=True)
    print(f"\n{len(frame)} rows -> {path}", flush=True)
    return path


def cmd_host(args, models: list[Model]) -> None:
    rows = []
    for model in models:
        parsed = [parse_medium(model, text) for text in args.medium]
        fitness = model.host([pairs for pairs, _ in parsed])
        for (pairs, labels), value in zip(parsed, fitness, strict=True):
            rows.append(
                {
                    "checkpoint": model.name,
                    "fold": model.fold["fold"],
                    "seed": model.seed,
                    "medium": " + ".join(labels) or "compound free",
                    "n_compounds": len(pairs),
                    "predicted_fitness": value,
                    "all_compounds_fitted": all(
                        "(unfitted)" not in label for label in labels
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if args.mean:
        frame = (
            frame.groupby(["medium", "n_compounds"], as_index=False)
            .agg(
                predicted_fitness=("predicted_fitness", "mean"),
                sd=("predicted_fitness", "std"),
                n_checkpoints=("predicted_fitness", "size"),
            )
            .sort_values("predicted_fitness", ascending=False, ignore_index=True)
        )
    write(frame, args.name or "host")


def cmd_grid(args, models: list[Model]) -> None:
    """The predicted fitness surface of two compounds, its boundary and its islands."""
    first, second = (c.strip() for c in args.pair.split(","))
    if args.run:
        wells = pd.read_csv(WELLS_CSV)
        grid = isobole_grid(wells, args.run)
        x_doses_g, a_doses_g = list(grid.x_doses), list(grid.a_doses)
        observed = np.array(grid.y).mean(axis=2)
        observed_grew = np.array(grid.g).all(axis=2)
        tau = grid.tau
        # the wet-lab table doses are g/L; the model reads mM, so convert by molar mass
        weight = {name: molar_mass(name) for name in (first, second)}
        x_doses = [d / weight[first] * 1e3 for d in x_doses_g]
        a_doses = [d / weight[second] * 1e3 for d in a_doses_g]
    else:
        assert args.doses and args.doses_second, "--doses and --doses-second, in mM"
        x_doses = [float(v) for v in args.doses.split(",")]
        a_doses = [float(v) for v in args.doses_second.split(",")]
        x_doses_g, a_doses_g = x_doses, a_doses
        observed = observed_grew = None
        tau = args.tau

    rows, surfaces = [], []
    for model in models:
        media, labels = [], []
        for i, x in enumerate(x_doses):
            for j, a in enumerate(a_doses):
                parts = []
                if x > 0:
                    parts.append(f"{first}={x}")
                if a > 0:
                    parts.append(f"{second}={a}")
                pairs, _ = parse_medium(model, ",".join(parts))
                media.append(pairs)
                labels.append((i, j))
        fitness = model.host(media)
        surface = np.zeros((len(x_doses), len(a_doses)))
        for (i, j), value in zip(labels, fitness, strict=True):
            surface[i, j] = value
        surfaces.append(surface)
        rows.extend(
            {
                "checkpoint": model.name,
                "fold": model.fold["fold"],
                "seed": model.seed,
                f"dose_{first}": x_doses_g[i],
                f"dose_{second}": a_doses_g[j],
                "predicted_fitness": surface[i, j],
            }
            for i in range(len(x_doses))
            for j in range(len(a_doses))
        )
    stem = args.name or f"grid_{args.run or 'custom'}"
    cells = pd.DataFrame(rows)
    write(cells, f"{stem}_cells")

    mean = np.mean(surfaces, axis=0)
    alive = mean >= tau
    closed = monotone_alive(alive)
    summary = {
        "run": args.run or "custom",
        "first": first,
        "second": second,
        "tau": tau,
        "checkpoints": len(models),
        "predicted_alive": int(alive.sum()),
        "predicted_alive_after_closure": int(closed.sum()),
        "island_cells_dropped": int((alive & ~closed).sum()),
    }
    if observed is not None:
        obs_alive = observed_grew[1:, 1:]
        for name, mask in (("raw", alive[1:, 1:]), ("monotone", closed[1:, 1:])):
            union = (obs_alive | mask).sum()
            summary[f"jaccard_{name}"] = (
                float((obs_alive & mask).sum() / union) if union else float("nan")
            )
            summary[f"alive_not_observed_{name}"] = int((mask & ~obs_alive).sum())
            summary[f"observed_not_alive_{name}"] = int((obs_alive & ~mask).sum())
        summary["mean_observed_minus_predicted"] = float(
            np.mean(observed[1:, 1:] - mean[1:, 1:])
        )
    write(pd.DataFrame([summary]), f"{stem}_summary")


def cmd_profile(args, models: list[Model]) -> None:
    """The gene-level response under one medium, over a named gene set."""
    rows = []
    for model in models:
        source = args.source or model.sources[0]["name"]
        panel = list(model.source_of[source]["genes"])
        if args.genes:
            genes = [g.strip() for g in args.genes.split(",")]
        elif args.genes_outside_panel:
            inside = set(panel)
            genes = [g for g in model.node_ids if g not in inside]
        else:
            genes = panel
        pairs, labels = parse_medium(model, args.medium)
        response = model.gene_response([[g] for g in genes], pairs, source)
        fitted = model.fitted_genes
        rows.append(
            pd.DataFrame(
                {
                    "checkpoint": model.name,
                    "fold": model.fold["fold"],
                    "seed": model.seed,
                    "source": source,
                    "medium": " + ".join(labels) or "compound free",
                    "gene": genes,
                    "predicted_response": response,
                    "in_source_panel": [g in set(panel) for g in genes],
                    "gene_fitted": [g in fitted for g in genes],
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    if args.mean:
        frame = (
            frame.groupby(
                ["source", "medium", "gene", "in_source_panel", "gene_fitted"],
                as_index=False,
            )
            .agg(
                predicted_response=("predicted_response", "mean"),
                sd=("predicted_response", "std"),
                n_checkpoints=("predicted_response", "size"),
            )
            .sort_values("predicted_response", ignore_index=True)
        )
    if args.top:
        frame = pd.concat([frame.head(args.top), frame.tail(args.top)])
    write(frame, args.name or "profile")


def cmd_genotype(args, models: list[Model]) -> None:
    """One genotype of any size under one medium."""
    genotypes = [[g.strip() for g in spec.split(",")] for spec in args.genotype]
    rows = []
    for model in models:
        source = args.source or model.sources[0]["name"]
        pairs, labels = parse_medium(model, args.medium)
        response = model.gene_response(genotypes, pairs, source)
        for genes, value in zip(genotypes, response, strict=True):
            rows.append(
                {
                    "checkpoint": model.name,
                    "fold": model.fold["fold"],
                    "seed": model.seed,
                    "source": source,
                    "medium": " + ".join(labels) or "compound free",
                    "genotype": " ".join(genes),
                    "order": len(genes),
                    "predicted_response": value,
                    "every_gene_fitted": all(g in model.fitted_genes for g in genes),
                }
            )
    frame = pd.DataFrame(rows)
    if args.mean:
        frame = frame.groupby(
            ["source", "medium", "genotype", "order", "every_gene_fitted"],
            as_index=False,
        ).agg(
            predicted_response=("predicted_response", "mean"),
            sd=("predicted_response", "std"),
            n_checkpoints=("predicted_response", "size"),
        )
    write(frame, args.name or "genotype")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list the checkpoints")
    sub = parser.add_subparsers(dest="command")
    for name in ("host", "grid", "profile", "genotype"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", nargs="*", default=None)
        p.add_argument("--arm", default=None, help="every checkpoint of this sweep")
        p.add_argument("--mean", action="store_true", help="average over checkpoints")
        p.add_argument("--name", default=None, help="output file stem")
        p.add_argument("--source", default=None)
        if name == "host":
            p.add_argument("--medium", action="append", required=True)
        if name == "grid":
            p.add_argument("--pair", required=True, help="first,second")
            p.add_argument("--run", default=None, choices=sorted(ISOBOLES))
            p.add_argument("--doses", default=None, help="mM of the first, comma list")
            p.add_argument("--doses-second", default=None, help="mM of the second")
            p.add_argument("--tau", type=float, default=0.1)
        if name == "profile":
            p.add_argument("--medium", required=True)
            p.add_argument("--genes", default=None, help="comma list, else the panel")
            p.add_argument("--genes-outside-panel", action="store_true")
            p.add_argument("--top", type=int, default=0)
        if name == "genotype":
            p.add_argument("--medium", required=True)
            p.add_argument("--genotype", action="append", required=True)
    args = parser.parse_args()

    if args.list or args.command is None:
        paths = sorted(glob(osp.join(CHECKPOINTS, "*", "*.pt")))
        frame = pd.DataFrame(
            [
                {
                    "sweep": osp.basename(osp.dirname(p)),
                    "checkpoint": osp.basename(p),
                    "megabytes": round(osp.getsize(p) / 2**20, 1),
                }
                for p in paths
            ]
        )
        print(
            frame.to_string(index=False)
            if len(frame)
            else f"no checkpoint under {CHECKPOINTS}",
            flush=True,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args, device)
    print(
        f"{len(models)} checkpoint(s): "
        + ", ".join(f"{m.name} (step {m.selected_step})" for m in models),
        flush=True,
    )
    {
        "host": cmd_host,
        "grid": cmd_grid,
        "profile": cmd_profile,
        "genotype": cmd_genotype,
    }[args.command](args, models)


if __name__ == "__main__":
    main()

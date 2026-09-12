# experiments/028-knockout-expression/scripts/nadal_assignment_purity.R
# [[experiments.028-knockout-expression.scripts.nadal_assignment_purity]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_assignment_purity
#
# How pure is the cell-to-genotype assignment in Nadal-Ribelles 2025 (control)? A cell
# truly carrying deletion g cannot express g. For every genotype whose deleted ORF is in
# the count matrix, this reports the fraction of its cells with at least one UMI of that
# ORF, beside the fraction of WT cells with at least one UMI of the same ORF. If
# assignment were clean, the first is near zero wherever the second is not; the ratio of
# the two is a per-genotype upper bound on the share of cells that are not that genotype
# (an estimate, since a retained 3' UTR could also give reads from a deleted locus).
#
# The sharper estimate is the EXCESS-ZERO purity. Among a genotype's cells, the fraction
# with zero UMI of the deleted gene is z_g; among WT cells it is z_wt. If the genotype's
# cells are a mixture of a fraction p of true deletion cells (always zero) and 1 - p of
# cells expressing the gene like WT, then z_g = p + (1 - p) z_wt, so
#     p = (z_g - z_wt) / (1 - z_wt).
# A uniformly reduced signal (a retained 3' UTR read through at a lower rate) gives
# instead a Poisson zero fraction near exp(-mean) with no excess beyond it; the per-cell
# distributions of the twelve most expressed deleted genes are bimodal, not uniformly
# reduced, which is what licenses the mixture reading. Reported for genotypes whose
# deleted gene WT detects in at least WT_DETECT of cells, where z_wt is small enough
# for the estimate to have resolution.
#
#   ~/miniconda3/envs/r-seurat/bin/Rscript experiments/028-knockout-expression/scripts/nadal_assignment_purity.R
suppressMessages({ library(Seurat); library(Matrix) })
raw <- Sys.getenv("NADAL_RAW", "/scratch/projects/torchcell-scratch/torchcell-raw/nadalRibelles2025")
data_root <- Sys.getenv("DATA_ROOT", "/scratch/projects/torchcell-scratch")
out_dir <- file.path(data_root, "data/torchcell/nadal_ribelles_perturbseq2025/recomputed")

load(file.path(raw, "seus_split.RData"))
ctrl <- seus$Control; rm(seus); invisible(gc())
md <- ctrl[[]]
label <- as.character(md$assignment_consensus2)
kogene <- as.character(md$kogene)
kosym <- as.character(md$kosym)
counts <- LayerData(ctrl[["RNA"]], layer = "counts")
genes <- rownames(counts)
detected <- counts
detected@x[] <- 1  # any UMI -> 1

groups <- sort(unique(label))
G <- sparseMatrix(i = seq_along(label), j = match(label, groups), x = 1,
                  dims = c(length(label), length(groups)), dimnames = list(NULL, groups))
n_cells <- as.integer(colSums(G)); names(n_cells) <- groups
det <- as.matrix(detected %*% G)                # genes x genotypes: cells with >= 1 UMI
det_frac <- sweep(det, 2, n_cells, "/")
umi <- as.matrix(counts %*% G)

# The deleted gene's row: try the systematic name, then the symbol, as row names mix both.
rows <- list()
for (g in groups) {
  if (g == "WT") next
  ko <- unique(kogene[label == g]); ko <- ko[!is.na(ko)]
  sym <- unique(kosym[label == g]); sym <- sym[!is.na(sym)]
  if (length(ko) != 1) next
  r <- match(ko, genes); if (is.na(r) && length(sym) == 1) r <- match(sym, genes)
  if (is.na(r)) next
  rows[[g]] <- data.frame(label = g, kogene = ko, n_cells = n_cells[g],
                          frac_cells_detecting_ko = det_frac[r, g],
                          wt_frac_detecting = det_frac[r, "WT"],
                          umi_ko_in_genotype = umi[r, g], umi_ko_in_wt = umi[r, "WT"],
                          umi_per_cell_ko = umi[r, g] / n_cells[g],
                          wt_umi_per_cell = umi[r, "WT"] / n_cells["WT"])
}
df <- do.call(rbind, rows)
df$impurity_upper_bound <- pmin(1, df$frac_cells_detecting_ko / pmax(df$wt_frac_detecting, 1e-9))
z_g <- 1 - df$frac_cells_detecting_ko
z_wt <- 1 - df$wt_frac_detecting
df$purity_excess_zero <- pmax(0, pmin(1, (z_g - z_wt) / pmax(1 - z_wt, 1e-9)))
WT_DETECT <- 0.5
write.table(df, file.path(out_dir, "assignment_purity.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("genotypes scored:", nrow(df), "\n")
ok <- df$wt_frac_detecting >= 0.5
cat("of which the deleted gene is detected in >= 50% of WT cells:", sum(ok), "\n")
cat("  among those, fraction of the genotype's own cells still detecting it: median",
    round(median(df$frac_cells_detecting_ko[ok]), 3), " IQR",
    paste(round(quantile(df$frac_cells_detecting_ko[ok], c(.25, .75)), 3), collapse = " .. "), "\n")
cat("  impurity upper bound (ratio to WT detection): median", round(median(df$impurity_upper_bound[ok]), 3),
    " IQR", paste(round(quantile(df$impurity_upper_bound[ok], c(.25, .75)), 3), collapse = " .. "), "\n")
cat("  excess-zero purity (share of true deletion cells): median", round(median(df$purity_excess_zero[ok]), 3),
    " IQR", paste(round(quantile(df$purity_excess_zero[ok], c(.25, .75)), 3), collapse = " .. "), "\n")
ok2 <- df$wt_frac_detecting >= 0.2
cat("at WT detection >= 0.2 (n =", sum(ok2), "): excess-zero purity median", round(median(df$purity_excess_zero[ok2]), 3),
    " IQR", paste(round(quantile(df$purity_excess_zero[ok2], c(.25, .75)), 3), collapse = " .. "), "\n")
cat("  genotypes with ratio < 0.1 (clean):", sum(df$impurity_upper_bound[ok] < 0.1),
    "  0.1-0.5:", sum(df$impurity_upper_bound[ok] >= 0.1 & df$impurity_upper_bound[ok] < 0.5),
    "  >= 0.5 (indistinguishable from WT):", sum(df$impurity_upper_bound[ok] >= 0.5), "\n")

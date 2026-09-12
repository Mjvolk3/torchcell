# experiments/028-knockout-expression/scripts/nadal_split_half_signal.R
# [[experiments.028-knockout-expression.scripts.nadal_split_half_signal]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_split_half_signal
#
# Do a genotype's cells carry ANY reproducible signal that distinguishes them from wild
# type, independent of Kemmeren? Split the genotype's cells at random into halves A and
# B. From A against the WT cells, derive a signature: the N_SIG genes with the largest
# |pseudobulk log2 FC| (summed UMI, CPM, pseudocount 1, at least MIN_UMI summed UMI), split
# by sign. Score every cell of B and every WT cell on that signature (mean log-normalized
# expression over up genes minus over down genes) and report the AUROC between B and WT.
# High AUROC = the labeled cells share a consistent transcriptome that is not WT, whatever
# it is; AUROC near 0.5 = no reproducible genotype-level signal at this depth. The same
# score with Kemmeren's signature (nadal_signature_mixture.R) is the comparison.
#
# The WT cells are split too: WT_REF (half) is the reference the signature is derived
# against and WT_TEST (the other half) is what B is scored against, so no cell on either
# side of the AUROC was used to choose the genes. Null: a random set of WT cells the size
# of a genotype plays the genotype (half derives, half is scored against WT_TEST), 40
# draws; expected 0.5.
#
#   ~/miniconda3/envs/r-seurat/bin/Rscript experiments/028-knockout-expression/scripts/nadal_split_half_signal.R
suppressMessages({ library(Seurat); library(Matrix) })
set.seed(0)
N_SIG <- 50; MIN_UMI <- 20; MIN_CELLS <- 40
raw <- Sys.getenv("NADAL_RAW", "/scratch/projects/torchcell-scratch/torchcell-raw/nadalRibelles2025")
data_root <- Sys.getenv("DATA_ROOT", "/scratch/projects/torchcell-scratch")
out_dir <- file.path(data_root, "data/torchcell/nadal_ribelles_perturbseq2025/recomputed")
sig_tab <- read.delim(file.path(out_dir, "signature_mixture.tsv"), stringsAsFactors = FALSE)

load(file.path(raw, "seus_split.RData"))
ctrl <- seus$Control; rm(seus); invisible(gc())
md <- ctrl[[]]
label <- as.character(md$assignment_consensus2)
kogene <- as.character(md$kogene)
counts <- LayerData(ctrl[["RNA"]], layer = "counts")
data <- LayerData(ctrl[["RNA"]], layer = "data")
genes <- rownames(counts)
wt <- which(label == "WT")
wt_sh <- sample(wt); wt_ref <- wt_sh[seq_len(length(wt_sh) %/% 2)]; wt_test <- setdiff(wt_sh, wt_ref)

auroc <- function(a, b) { r <- rank(c(a, b)); (sum(r[seq_along(a)]) - length(a) * (length(a) + 1) / 2) / (length(a) * length(b)) }
signature <- function(idx_a, idx_ref) {
  pa <- Matrix::rowSums(counts[, idx_a, drop = FALSE]); pr <- Matrix::rowSums(counts[, idx_ref, drop = FALSE])
  ok <- (pa + pr) >= MIN_UMI
  lfc <- log2(pa / sum(pa) * 1e6 + 1) - log2(pr / sum(pr) * 1e6 + 1)
  lfc[!ok] <- NA
  top <- order(-abs(lfc), na.last = NA)[seq_len(min(N_SIG, sum(ok)))]
  list(up = genes[top][lfc[top] > 0], down = genes[top][lfc[top] < 0])
}
score <- function(s, idx) {
  u <- if (length(s$up)) Matrix::colMeans(data[s$up, idx, drop = FALSE]) else 0
  d <- if (length(s$down)) Matrix::colMeans(data[s$down, idx, drop = FALSE]) else 0
  as.numeric(u - d)
}
# Genotypes: the ones scored against Kemmeren (so the two AUROCs are on the same set), with enough cells.
rows <- list()
for (ko in sig_tab$kogene) {
  cells <- which(kogene == ko & label != "WT")
  if (length(cells) < MIN_CELLS) next
  h <- sample(cells); a <- h[seq_len(length(h) %/% 2)]; b <- setdiff(h, a)
  s <- signature(a, wt_ref)
  rows[[ko]] <- data.frame(kogene = ko, n_cells = length(cells),
                           auroc_split_half = auroc(score(s, b), score(s, wt_test)),
                           auroc_kemmeren = sig_tab$auroc[sig_tab$kogene == ko][1])
}
df <- do.call(rbind, rows)
# Null: WT cells playing a genotype of median size, held-out reference and test.
n_med <- as.integer(median(df$n_cells))
ctrl_auc <- replicate(40, {
  g <- sample(wt_ref, min(n_med, length(wt_ref) - 50)); a <- g[seq_len(length(g) %/% 2)]; b <- setdiff(g, a)
  s <- signature(a, setdiff(wt_ref, g)); auroc(score(s, b), score(s, wt_test)) })
write.table(df, file.path(out_dir, "split_half_signal.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("genotypes:", nrow(df), "(>=", MIN_CELLS, "cells)\n")
cat("split-half AUROC (own signature): median", round(median(df$auroc_split_half), 3), " IQR", paste(round(quantile(df$auroc_split_half, c(.25, .75)), 3), collapse = " .. "), "\n")
cat("Kemmeren-signature AUROC, same genotypes: median", round(median(df$auroc_kemmeren), 3), "\n")
cat("genotypes with split-half AUROC > 0.8:", sum(df$auroc_split_half > 0.8), "  > 0.65:", sum(df$auroc_split_half > 0.65), "  <= 0.55:", sum(df$auroc_split_half <= 0.55), "\n")
cat("null (WT cells as a genotype, held-out reference and test) AUROC: median", round(median(ctrl_auc), 3), " range", paste(round(range(ctrl_auc), 3), collapse = " .. "), "\n")
cat("Spearman(split-half AUROC, Kemmeren AUROC):", round(cor(df$auroc_split_half, df$auroc_kemmeren, method = "spearman"), 3), "\n")

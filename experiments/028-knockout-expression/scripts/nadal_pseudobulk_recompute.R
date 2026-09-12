# experiments/028-knockout-expression/scripts/nadal_pseudobulk_recompute.R
# [[experiments.028-knockout-expression.scripts.nadal_pseudobulk_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_pseudobulk_recompute
#
# Recompute the per-genotype fold change of Nadal-Ribelles 2025 (control condition) from
# the single-cell object, three ways, so the comparison against the microarrays can tell
# a bad statistic from bad sampling.
#
# Input: seus_split.RData (Zenodo 14062629, md5 65bb56efd8120f32f65c044de5f040aa), the
# `Control` Seurat object: RNA assay `counts` (raw UMI) and `data` (Seurat log-normalized,
# log1p of counts scaled to 1e4 per cell), genotype label `assignment_consensus2` (the
# `bc-<ORF>` barcode label, `WT` for the wild-type clones), `kogene` the deleted ORF.
#
# Three statistics per (genotype, gene), each against the pooled WT cells of the same
# condition:
#
#   A  pseudobulk_log2fc   bulk-like. Sum raw UMIs over the genotype's cells and over the
#                          WT cells, scale each to counts per million, then
#                          log2((cpm_g + PRIOR) / (cpm_wt + PRIOR)). This is the quantity a
#                          two-color microarray log2 ratio is comparable to. A gene is
#                          reported for a genotype only when the summed UMI over the
#                          genotype's cells and the WT cells is at least MIN_UMI; below
#                          that the ratio is a ratio of a handful of molecules and is left
#                          absent, which is the same honesty convention the loader uses
#                          for genome-absent genes.
#   B  seurat_avg_log2fc   Seurat's FindMarkers default: log2(mean_cells(expm1(data_g)) + 1)
#                          - log2(mean_cells(expm1(data_wt)) + 1), a per-cell mean with a
#                          pseudocount of one.
#   C  scanpy_logfc        scanpy's rank_genes_groups formula on the same per-cell means,
#                          log2((expm1(mean(data_g)) + 1e-9) / (expm1(mean(data_wt)) + 1e-9)),
#                          computed here only to check that it reproduces the stored
#                          values and their +-23 sentinels, i.e. that the stored numbers
#                          are this statistic.
#
# Output: one gzipped TSV per statistic, genes x genotypes, plus a genotype table with the
# deleted ORF and the cell count, under
# $DATA_ROOT/data/torchcell/nadal_ribelles_perturbseq2025/recomputed/.
#
#   ~/miniconda3/envs/r-seurat/bin/Rscript experiments/028-knockout-expression/scripts/nadal_pseudobulk_recompute.R
suppressMessages({ library(Seurat); library(Matrix) })

PRIOR <- 1      # CPM added to both sides of the bulk-like ratio
MIN_UMI <- 10   # summed UMI (genotype + WT) below which a gene is left absent

raw <- Sys.getenv("NADAL_RAW", "/scratch/projects/torchcell-scratch/torchcell-raw/nadalRibelles2025")
data_root <- Sys.getenv("DATA_ROOT", "/scratch/projects/torchcell-scratch")
out_dir <- file.path(data_root, "data/torchcell/nadal_ribelles_perturbseq2025/recomputed")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

t0 <- Sys.time()
load(file.path(raw, "seus_split.RData"))
ctrl <- seus$Control
rm(seus); invisible(gc())
cat("loaded Control:", ncol(ctrl), "cells,", nrow(ctrl[["RNA"]]), "genes in", format(Sys.time() - t0), "\n")

md <- ctrl[[]]
label <- as.character(md$assignment_consensus2)
keep <- !is.na(label) & label != ""
stopifnot(any(label == "WT"))
counts <- LayerData(ctrl[["RNA"]], layer = "counts")[, keep]
data <- LayerData(ctrl[["RNA"]], layer = "data")[, keep]
label <- label[keep]
kogene <- as.character(md$kogene)[keep]
genes <- rownames(counts)
cat("cells with a genotype label:", ncol(counts), " WT cells:", sum(label == "WT"), "\n")

# Cell x genotype indicator, so every per-genotype sum is one sparse product.
groups <- sort(unique(label))
G <- sparseMatrix(i = seq_along(label), j = match(label, groups), x = 1,
                  dims = c(length(label), length(groups)), dimnames = list(NULL, groups))
n_cells <- as.integer(colSums(G))
names(n_cells) <- groups

# A: pseudobulk counts and CPM.
pb <- as.matrix(counts %*% G)                       # genes x genotypes, summed UMI
lib <- colSums(pb)
cpm <- sweep(pb, 2, lib / 1e6, "/")
wt_cpm <- cpm[, "WT"]
wt_pb <- pb[, "WT"]
A <- log2(sweep(cpm, 1, 0, "+") + PRIOR) - log2(wt_cpm + PRIOR)
absent <- sweep(pb, 1, wt_pb, "+") < MIN_UMI
A[absent] <- NA

# B and C: per-cell means of the de-logged normalized values.
ex <- data
ex@x <- expm1(ex@x)
mean_ex <- as.matrix(ex %*% G)
mean_ex <- sweep(mean_ex, 2, n_cells, "/")
wt_mean <- mean_ex[, "WT"]
B <- log2(mean_ex + 1) - log2(wt_mean + 1)
C <- log2(mean_ex + 1e-9) - log2(wt_mean + 1e-9)

# Genotype table: label, deleted ORF, cells.
ko <- tapply(kogene, label, function(v) { u <- unique(v[!is.na(v)]); if (length(u) == 1) u else NA_character_ })
geno <- data.frame(label = groups, kogene = as.character(ko[groups]), n_cells = n_cells[groups],
                   stringsAsFactors = FALSE)

write_mat <- function(m, name) {
  f <- gzfile(file.path(out_dir, paste0(name, ".tsv.gz")), "w")
  write.table(data.frame(gene = genes, m, check.names = FALSE), f, sep = "\t", quote = FALSE, row.names = FALSE)
  close(f)
  cat("wrote", name, dim(m)[1], "x", dim(m)[2], "\n")
}
write_mat(A, "pseudobulk_log2fc")
write_mat(B, "seurat_avg_log2fc")
write_mat(C, "scanpy_logfc")
write.table(geno, file.path(out_dir, "genotypes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("genotypes:", nrow(geno), " with a single deleted ORF:", sum(!is.na(geno$kogene)), "\n")
cat("A: cells reported", sum(!is.na(A)), "of", length(A), "; sd", round(sd(A, na.rm = TRUE), 3), "\n")
cat("B: sd", round(sd(B), 3), "   C: sd", round(sd(C), 3), " frac |C| >= 20:", round(mean(abs(C) >= 20), 3), "\n")
cat("done in", format(Sys.time() - t0), "\n")

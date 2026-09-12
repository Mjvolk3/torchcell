# experiments/028-knockout-expression/scripts/nadal_batch_replication.R
# [[experiments.028-knockout-expression.scripts.nadal_batch_replication]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_batch_replication
#
# How reproducible is a Nadal-Ribelles 2025 (control) genotype profile WITHIN the study?
# Kemmeren against Sameith gives the microarray platform's own test-retest (per-strain
# median r 0.74); nothing equivalent exists for the single-cell panel, and without it a
# cross-study agreement near zero cannot be read. Two internal replications, from the
# raw UMI of `seus_split.RData`:
#
#   cross-batch    the same genotype's cells fall in several Singleron GEXSCOPE cartridge
#                  batches (14 control cartridges,
#                  WT cells in every one). Pseudobulk log2 FC per (genotype, batch)
#                  against the WT cells OF THE SAME BATCH, then the Pearson r between two
#                  batches of the same genotype on genes with at least MIN_UMI summed UMI
#                  on both sides. Null: two batches of two DIFFERENT genotypes.
#   split-half     the same genotype's cells within one batch, split at random into two
#                  halves, each against the batch's WT cells; r between the halves.
#
# Both are also computed against the POOLED WT (all batches) so the batch effect itself
# is measurable: the same-genotype cross-batch r with a pooled reference minus the r with
# a per-batch reference is what batch contributes; WT-vs-WT between batches is the
# batch effect on its own.
#
# Also written: per-cell depth (UMI and genes detected per cell) and per-genotype means
# of every numeric metadata column, so any per-cell score the authors stored (ESR and
# the like) is available at genotype level.
#
# Output under $DATA_ROOT/data/torchcell/nadal_ribelles_perturbseq2025/recomputed/:
#   batch_replication_pairs.tsv     one row per (genotype, batch i, batch j) pair
#   batch_replication_null.tsv      different-genotype pairs, cell counts matched by bin
#   split_half_pairs.tsv            one row per (genotype, batch) split
#   wt_batch_effect.tsv             WT batch i vs WT batch j
#   cell_depth.tsv                  per-cell UMI, genes detected, batch, genotype
#   genotype_meta_means.tsv         per-genotype means of numeric metadata columns
#
#   ~/miniconda3/envs/r-seurat/bin/Rscript experiments/028-knockout-expression/scripts/nadal_batch_replication.R
suppressMessages({ library(Seurat); library(Matrix) })
set.seed(0)
MIN_CELLS <- 15      # cells of a genotype in a batch for that batch to count
MIN_WT <- 15         # WT cells a batch needs to serve as a per-batch reference
MIN_UMI <- 20        # summed UMI (both sides) below which a gene is not compared
MIN_GENES <- 300     # genes compared below which a pair is not reported
PRIOR <- 1

raw <- Sys.getenv("NADAL_RAW", "/scratch/projects/torchcell-scratch/torchcell-raw/nadalRibelles2025")
data_root <- Sys.getenv("DATA_ROOT", "/scratch/projects/torchcell-scratch")
out_dir <- file.path(data_root, "data/torchcell/nadal_ribelles_perturbseq2025/recomputed")

t0 <- Sys.time()
load(file.path(raw, "seus_split.RData"))
ctrl <- seus$Control; rm(seus); invisible(gc())
md <- ctrl[[]]
label <- as.character(md$assignment_consensus2)
batch <- as.character(md$batch)
keep <- !is.na(label) & label != "" & !is.na(batch)
counts <- LayerData(ctrl[["RNA"]], layer = "counts")[, keep]
label <- label[keep]; batch <- batch[keep]; md <- md[keep, ]
genes <- rownames(counts)
cat("cells", ncol(counts), "genes", length(genes), "batches", length(unique(batch)),
    "loaded in", format(Sys.time() - t0), "\n")

# ---- per-cell depth and per-genotype metadata means -------------------------------
depth <- data.frame(cell = colnames(counts), umi = colSums(counts),
                    genes_detected = colSums(counts > 0), batch = batch, genotype = label)
write.table(depth, file.path(out_dir, "cell_depth.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
num <- md[, sapply(md, is.numeric), drop = FALSE]
means <- aggregate(num, by = list(genotype = label), FUN = mean)
means$n_cells <- as.integer(table(label)[means$genotype])
write.table(means, file.path(out_dir, "genotype_meta_means.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("metadata columns averaged:", paste(colnames(num), collapse = ", "), "\n")

# ---- pseudobulk per (genotype, batch) ---------------------------------------------
key <- paste(label, batch, sep = "|")
groups <- sort(unique(key))
G <- sparseMatrix(i = seq_along(key), j = match(key, groups), x = 1,
                  dims = c(length(key), length(groups)), dimnames = list(NULL, groups))
pb <- as.matrix(counts %*% G)                       # genes x (genotype|batch)
n_cells <- as.integer(colSums(G)); names(n_cells) <- groups
gb_geno <- sub("\\|.*$", "", groups)
gb_batch <- sub("^.*\\|", "", groups)
cpm <- sweep(pb, 2, colSums(pb) / 1e6, "/")

wt_cols <- groups[gb_geno == "WT"]
wt_batch_ok <- gb_batch[gb_geno == "WT"][n_cells[wt_cols] >= MIN_WT]
cat("batches with >=", MIN_WT, "WT cells:", length(wt_batch_ok), "of", length(unique(batch)), "\n")
wt_pool_pb <- rowSums(pb[, wt_cols, drop = FALSE])
wt_pool_cpm <- wt_pool_pb / sum(wt_pool_pb) * 1e6

log2fc <- function(col, ref_cpm, ref_pb) {
  ok <- (pb[, col] + ref_pb) >= MIN_UMI
  v <- log2(cpm[, col] + PRIOR) - log2(ref_cpm + PRIOR)
  v[!ok] <- NA
  v
}
pair_r <- function(a, b) {
  ok <- is.finite(a) & is.finite(b)
  if (sum(ok) < MIN_GENES) return(c(NA, sum(ok)))
  c(cor(a[ok], b[ok]), sum(ok))
}

# Every (genotype, batch) with enough cells, in a batch with a WT reference.
elig <- groups[gb_geno != "WT" & n_cells >= MIN_CELLS & gb_batch %in% wt_batch_ok]
cat("eligible (genotype, batch) pseudobulks:", length(elig), "\n")
fc_batch <- sapply(elig, function(col) {
  b <- sub("^.*\\|", "", col); w <- paste("WT", b, sep = "|")
  log2fc(col, cpm[, w], pb[, w])
})
fc_pool <- sapply(elig, function(col) log2fc(col, wt_pool_cpm, wt_pool_pb))
e_geno <- sub("\\|.*$", "", elig); e_batch <- sub("^.*\\|", "", elig)

# ---- same-genotype cross-batch pairs -------------------------------------------------
rows <- list()
for (g in unique(e_geno[duplicated(e_geno)])) {
  idx <- which(e_geno == g)
  for (i in idx) for (j in idx) if (i < j) {
    rb <- pair_r(fc_batch[, i], fc_batch[, j]); rp <- pair_r(fc_pool[, i], fc_pool[, j])
    rows[[length(rows) + 1]] <- data.frame(
      genotype = g, batch_i = e_batch[i], batch_j = e_batch[j],
      cells_i = n_cells[elig[i]], cells_j = n_cells[elig[j]],
      r_batch_ref = rb[1], r_pool_ref = rp[1], n_genes = rb[2])
  }
}
same <- do.call(rbind, rows)
write.table(same, file.path(out_dir, "batch_replication_pairs.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("same-genotype cross-batch pairs:", nrow(same), " median r (batch ref)",
    round(median(same$r_batch_ref, na.rm = TRUE), 3), " (pool ref)",
    round(median(same$r_pool_ref, na.rm = TRUE), 3), "\n")

# ---- null: different genotypes, different batches, cell counts matched by bin --------
cbin <- cut(n_cells[elig], c(0, 30, 60, 120, 1e9))
null_rows <- list()
for (k in seq_len(nrow(same))) {
  i <- which(elig == paste(same$genotype[k], same$batch_i[k], sep = "|"))
  cand <- which(e_geno != same$genotype[k] & e_batch == same$batch_j[k] & cbin == cbin[i])
  if (!length(cand)) cand <- which(e_geno != same$genotype[k] & e_batch == same$batch_j[k])
  j <- sample(cand, 1)
  rb <- pair_r(fc_batch[, i], fc_batch[, j]); rp <- pair_r(fc_pool[, i], fc_pool[, j])
  null_rows[[k]] <- data.frame(genotype_i = e_geno[i], genotype_j = e_geno[j],
                               batch_i = e_batch[i], batch_j = e_batch[j],
                               cells_i = n_cells[elig[i]], cells_j = n_cells[elig[j]],
                               r_batch_ref = rb[1], r_pool_ref = rp[1], n_genes = rb[2])
}
null <- do.call(rbind, null_rows)
write.table(null, file.path(out_dir, "batch_replication_null.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("different-genotype pairs:", nrow(null), " median r (batch ref)",
    round(median(null$r_batch_ref, na.rm = TRUE), 3), " (pool ref)",
    round(median(null$r_pool_ref, na.rm = TRUE), 3), "\n")

# ---- split-half within (genotype, batch) ---------------------------------------------
# The WT cells of the batch are split too: half A of the genotype against WT half A,
# half B against WT half B. With one shared WT reference the two halves would share
# its sampling noise and correlate for that reason alone. Null: two DIFFERENT
# genotypes of the same batch, one against each WT half.
sh_rows <- list()
sh_elig <- elig[n_cells[elig] >= 2 * MIN_CELLS]
for (col in sh_elig) {
  g <- sub("\\|.*$", "", col); b <- sub("^.*\\|", "", col)
  wt_cells <- sample(which(key == paste("WT", b, sep = "|")))
  wa <- wt_cells[seq_len(length(wt_cells) %/% 2)]; wb <- wt_cells[-seq_len(length(wt_cells) %/% 2)]
  ra <- rowSums(counts[, wa, drop = FALSE]); rb <- rowSums(counts[, wb, drop = FALSE])
  ca <- ra / sum(ra) * 1e6; cb <- rb / sum(rb) * 1e6
  half_fc <- function(cells_idx, ref_cpm, ref_pb) {
    p <- rowSums(counts[, cells_idx, drop = FALSE])
    v <- log2(p / sum(p) * 1e6 + PRIOR) - log2(ref_cpm + PRIOR)
    v[(p + ref_pb) < MIN_UMI] <- NA
    v
  }
  cells <- sample(which(key == col))
  a <- cells[seq_len(length(cells) %/% 2)]; bb <- cells[-seq_len(length(cells) %/% 2)]
  r <- pair_r(half_fc(a, ca, ra), half_fc(bb, cb, rb))
  # Null partner: another eligible genotype of this batch, its own half against WT B.
  others <- sh_elig[sh_elig != col & sub("^.*\\|", "", sh_elig) == b]
  r0 <- c(NA, NA)
  if (length(others)) {
    oc <- sample(which(key == sample(others, 1)))
    ob <- oc[-seq_len(length(oc) %/% 2)]
    r0 <- pair_r(half_fc(a, ca, ra), half_fc(ob, cb, rb))
  }
  sh_rows[[length(sh_rows) + 1]] <- data.frame(genotype = g, batch = b, cells = n_cells[col],
                                               wt_cells = length(wt_cells),
                                               r_split_half = r[1], n_genes = r[2],
                                               r_null_other_genotype = r0[1])
}
sh <- do.call(rbind, sh_rows)
write.table(sh, file.path(out_dir, "split_half_pairs.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("split-half pairs:", nrow(sh), " median r", round(median(sh$r_split_half, na.rm = TRUE), 3),
    " null (other genotype)", round(median(sh$r_null_other_genotype, na.rm = TRUE), 3), "\n")

# ---- WT batch effect: WT batch i against WT batch j, as a log2 FC sd and r to pool ----
wb <- list()
wt_ok <- paste("WT", wt_batch_ok, sep = "|")
for (i in seq_along(wt_ok)) for (j in seq_along(wt_ok)) if (i < j) {
  a <- wt_ok[i]; b <- wt_ok[j]
  ok <- (pb[, a] + pb[, b]) >= MIN_UMI
  v <- log2(cpm[ok, a] + PRIOR) - log2(cpm[ok, b] + PRIOR)
  wb[[length(wb) + 1]] <- data.frame(batch_i = wt_batch_ok[i], batch_j = wt_batch_ok[j],
                                     cells_i = n_cells[a], cells_j = n_cells[b],
                                     sd_log2fc = sd(v), n_genes = sum(ok),
                                     frac_abs_gt1 = mean(abs(v) > 1))
}
wbe <- do.call(rbind, wb)
write.table(wbe, file.path(out_dir, "wt_batch_effect.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("WT batch pairs:", nrow(wbe), " median sd of log2 FC", round(median(wbe$sd_log2fc), 3), "\n")
cat("done in", format(Sys.time() - t0), "\n")

# experiments/028-knockout-expression/scripts/nadal_signature_mixture.R
# [[experiments.028-knockout-expression.scripts.nadal_signature_mixture]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_signature_mixture
#
# A purity test that never touches the deleted gene. For each Nadal-Ribelles control
# genotype whose deletion has a strong Kemmeren response (at least MIN_SIG reporters with
# |log2 ratio| > 1), score every cell of that genotype and every wild-type cell on the
# Kemmeren signature: mean log-normalized expression over the up reporters minus the mean
# over the down reporters. If every labeled cell is the genotype, the genotype's score
# distribution is shifted as a whole against WT; if the labeled cells are a mixture, it is
# WT-like with a shifted minority. Reported per genotype: the AUROC of the score between
# the genotype's cells and WT (1 = every cell separates, 0.5 = indistinguishable), the
# fraction of the genotype's cells above the WT 95th percentile, and the shift of the
# median in WT standard deviations.
#
#   ~/miniconda3/envs/r-seurat/bin/Rscript experiments/028-knockout-expression/scripts/nadal_signature_mixture.R
suppressMessages({ library(Seurat); library(Matrix) })
MIN_SIG <- 20
raw <- Sys.getenv("NADAL_RAW", "/scratch/projects/torchcell-scratch/torchcell-raw/nadalRibelles2025")
data_root <- Sys.getenv("DATA_ROOT", "/scratch/projects/torchcell-scratch")
out_dir <- file.path(data_root, "data/torchcell/nadal_ribelles_perturbseq2025/recomputed")
sig <- read.delim(file.path(out_dir, "kemmeren_signatures.tsv"), stringsAsFactors = FALSE)

load(file.path(raw, "seus_split.RData"))
ctrl <- seus$Control; rm(seus); invisible(gc())
md <- ctrl[[]]
label <- as.character(md$assignment_consensus2)
kogene <- as.character(md$kogene)
data <- LayerData(ctrl[["RNA"]], layer = "data")
genes <- rownames(data)
wt <- which(label == "WT")

auroc <- function(a, b) { r <- rank(c(a, b)); (sum(r[seq_along(a)]) - length(a) * (length(a) + 1) / 2) / (length(a) * length(b)) }
rows <- list()
for (ko in unique(sig$kogene)) {
  s <- sig[sig$kogene == ko, ]
  up <- intersect(s$gene[s$direction == "up"], genes); dn <- intersect(s$gene[s$direction == "down"], genes)
  if (length(up) + length(dn) < MIN_SIG) next
  cells <- which(kogene == ko & label != "WT")
  if (length(cells) < 20) next
  score <- function(idx) {
    u <- if (length(up)) Matrix::colMeans(data[up, idx, drop = FALSE]) else 0
    d <- if (length(dn)) Matrix::colMeans(data[dn, idx, drop = FALSE]) else 0
    as.numeric(u - d)
  }
  sg <- score(cells); sw <- score(wt)
  rows[[ko]] <- data.frame(kogene = ko, n_cells = length(cells), n_up = length(up), n_down = length(dn),
                           auroc = auroc(sg, sw),
                           frac_above_wt95 = mean(sg > quantile(sw, 0.95)),
                           median_shift_wt_sd = (median(sg) - median(sw)) / sd(sw))
}
df <- do.call(rbind, rows)
write.table(df, file.path(out_dir, "signature_mixture.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("genotypes scored:", nrow(df), "\n")
cat("AUROC median", round(median(df$auroc), 3), " IQR", paste(round(quantile(df$auroc, c(.25, .75)), 3), collapse = " .. "), "\n")
cat("fraction of own cells above WT 95th pct: median", round(median(df$frac_above_wt95), 3), " (0.05 = WT-like, 1 = all cells shifted)\n")
cat("median shift in WT sd: median", round(median(df$median_shift_wt_sd), 3), "\n")
cat("genotypes with AUROC > 0.8:", sum(df$auroc > 0.8), "  > 0.65:", sum(df$auroc > 0.65), "  <= 0.55:", sum(df$auroc <= 0.55), "\n")

# experiments/028-knockout-expression/scripts/nadal_paper_deleteome_comparison.R
# [[experiments.028-knockout-expression.scripts.nadal_paper_deleteome_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_paper_deleteome_comparison
#
# What Nadal-Ribelles 2025 itself computed against the Kemmeren 2014 deleteome, rerun
# from the released objects with the paper's own code path (Figures_Rev.R, block
# "Supp Fig 1 I-J"). Per genotype with more than 5 cells the paper computes
#   javsma   Spearman between the single-cell scanpy log fold change and the microarray
#            M (log2 mutant/WT) over the genes present in both,
#   MAsig    the number of microarray genes with |M| > 1 and p < 0.05,
#   JAnsig   the number of single-cell DEGs (their DEG.Rdata: p < 0.05 and |logFC| >= 1),
# then plots ONLY log(MAsig) against log(JAnsig) with a Spearman label (FigS1I). The
# per-genotype profile correlation `javsma` is computed and never shown. This script
# writes all three per genotype, with the cell count, so both the shown and the unshown
# statistic can be drawn.
#
# Also written from the deleteome table:
#   kemmeren_responsive.tsv   per mutant: genes at FC > 1.7 (|M| > log2 1.7) and p < 0.05,
#                             the paper's responsive call (>= 4 genes), and the total
#                             genes with a p value; the deleteome's own extra profiles
#                             (wt-ypd vs wt, wt-matA vs wt, wt-by4743 vs wt) as columns
#                             of deleteome_extra_profiles.tsv with M and p.
#
# Inputs (raw mirror nadalRibelles2025, md5-verified against Zenodo 14062629):
#   deleteome_all_mutants_controls.txt, DEG.Rdata, FC_genotype.Rdata
# and the genotypes table from nadal_pseudobulk_recompute.R (cell counts).
#
#   ~/miniconda3/envs/r-seurat/bin/Rscript experiments/028-knockout-expression/scripts/nadal_paper_deleteome_comparison.R
suppressMessages(library(dplyr))
raw <- Sys.getenv("NADAL_RAW", "/scratch/projects/torchcell-scratch/torchcell-raw/nadalRibelles2025")
data_root <- Sys.getenv("DATA_ROOT", "/scratch/projects/torchcell-scratch")
out_dir <- file.path(data_root, "data/torchcell/nadal_ribelles_perturbseq2025/recomputed")

# ---- deleteome table, parsed as the paper does ---------------------------------------
madata <- read.table(file.path(raw, "deleteome_all_mutants_controls.txt"), sep = "\t",
                     comment.char = "", header = FALSE, quote = "", stringsAsFactors = FALSE)
geneinfo <- madata[-(1:2), 1:3]
colnames(geneinfo) <- madata[1, 1:3]
madata <- madata[, -(1:3)]
pdo <- data.frame(mutant = as.character(madata[1, ]), type = as.character(madata[2, ]),
                  stringsAsFactors = FALSE)
pdo$ko <- sapply(strsplit(pdo$mutant, split = "-"), "[", 1)
rownames(pdo) <- paste0(pdo[, 1], "_", pdo[, 2])
madata <- madata[-(1:2), ]
colnames(madata) <- rownames(pdo)
rownames(madata) <- geneinfo$reporterId
cat("deleteome: reporters", nrow(madata), " profiles", length(unique(pdo$mutant)), "\n")

# Responsive call per profile (Kemmeren: FC > 1.7 and p < 0.05, responsive if >= 4 genes,
# after excluding the 58 WT-variable genes and YDL196W listed in the paper's Extended
# Experimental Procedures, "Statistical Analysis of Expression Profiles").
WT_VARIABLE <- toupper(c("AI1", "AI2", "AI4", "AI5_ALPHA", "AI5_BETA", "ATP8", "BIO3", "BIO4", "BIO5",
  "BSC1", "DDR2", "FIT2", "GLK1", "GSY1", "HSP12", "HSP30", "HSP42", "HXK1", "NCE103", "OLI1",
  "PHO84", "PRM7", "SOL4", "SPL2", "SRO9", "STP4", "TPS2", "TSL1", "VAR1", "VTC3", "YDL038C",
  "YDR170W-A", "YDR210C-C", "YIG1", "YJR154W", "YKR075C", "YNL284C-A", "YOR343W-B", "YRO2",
  "ZEO1", "AIM33", "CTR1", "GPD2", "GPH1", "PHO12", "PKH2", "RIF2", "RTC3", "RTC4", "VTC1",
  "YDL177C", "YDR210W-B", "YER053C-A", "YFL002W-B", "YMR046C", "YPR158W-A", "ZRT1", "YDL196W"))
excl <- toupper(geneinfo$geneSymbol) %in% WT_VARIABLE | toupper(geneinfo$systematicName) %in% WT_VARIABLE
cat("WT-variable reporters excluded from the responsive call:", sum(excl), "\n")
profiles <- unique(pdo$mutant)
resp <- do.call(rbind, lapply(profiles, function(m) {
  M <- as.numeric(madata[, paste0(m, "_M")]); p <- as.numeric(madata[, paste0(m, "_p_value")])
  ok <- is.finite(M) & is.finite(p)
  sig <- ok & abs(M) > log2(1.7) & p < 0.05
  n17 <- sum(sig & !excl)
  data.frame(mutant = m, ko = pdo$ko[match(m, pdo$mutant)], n_genes_with_p = sum(ok),
             n_sig_fc1p7 = n17, n_sig_fc1p7_all_reporters = sum(sig),
             n_sig_fc2 = sum(ok & abs(M) > 1 & p < 0.05 & !excl),
             responsive = n17 >= 4, sd_M = sd(M[ok]), stringsAsFactors = FALSE)
}))
resp$systematic <- geneinfo$systematicName[match(toupper(resp$ko), geneinfo$geneSymbol)]
write.table(resp, file.path(out_dir, "kemmeren_responsive.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("responsive (>= 4 genes at FC > 1.7, p < 0.05):", sum(resp$responsive), "of", nrow(resp), "\n")

# The extra profiles (media, mating type, ploidy) as one table.
extra <- profiles[!grepl("-del", profiles)]
cat("extra profiles:", paste(extra, collapse = "; "), "\n")
ex <- data.frame(reporterId = rownames(madata), systematicName = geneinfo$systematicName,
                 geneSymbol = geneinfo$geneSymbol, stringsAsFactors = FALSE)
for (m in extra) {
  ex[[paste0(make.names(m), "_M")]] <- as.numeric(madata[, paste0(m, "_M")])
  ex[[paste0(make.names(m), "_p")]] <- as.numeric(madata[, paste0(m, "_p_value")])
}
write.table(ex, file.path(out_dir, "deleteome_extra_profiles.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# ---- the paper's own comparison ----------------------------------------------------
load(file.path(raw, "DEG.Rdata"))          # DEG: genotype, condition, upregulated, downregulated, total
load(file.path(raw, "FC_genotype.Rdata"))  # fcs: per DEG csv, names + logfoldchanges
degs <- DEG %>% group_by(genotype, condition) %>%
  summarize(upregulated = sum(upregulated), downregulated = sum(downregulated), .groups = "drop")
degs <- degs[degs$condition == "Control", ]

aux <- grep("Control", names(fcs))
fcs2 <- fcs[aux]
gns <- unique(unlist(lapply(fcs2, "[", "names")))
fc <- lapply(fcs2, function(y) y$logfoldchanges[match(gns, y$names)])
fc <- do.call(cbind, fc)
rownames(fc) <- gns
colnames(fc) <- sub("DEG_Control_", "", sub(".csv", "", colnames(fc)))
colnames(fc) <- gsub("_", "-", colnames(fc))
cat("single-cell fold-change matrix:", nrow(fc), "genes x", ncol(fc), "genotypes\n")

geno <- read.delim(file.path(out_dir, "genotypes.tsv"), stringsAsFactors = FALSE)
ncells <- setNames(geno$n_cells, geno$label)

pd <- pdo
pd$kosym <- toupper(pd$ko)
pd$sys <- geneinfo$systematicName[match(pd$kosym, geneinfo$geneSymbol)]
pd$acon2 <- geno$label[match(pd$sys, geno$kogene)]
pd$ncells <- ncells[pd$acon2]
pd <- pd[!is.na(pd$acon2), ]
pdctrl <- pd[pd$acon2 %in% colnames(fc), ]
pdctrl <- pdctrl[pdctrl$ncells > 5, ]
sel <- intersect(colnames(fc), pdctrl$acon2)
cat("genotypes compared (paper's rule, > 5 cells, in both):", length(sel), "\n")

allcors <- lapply(sel, function(i) {
  j <- unique(pdctrl[pdctrl$acon2 == i, ]$mutant)[1]
  x <- data.frame(fc[, i, drop = FALSE]); x$gene <- rownames(x)
  # The paper selects the mutant's columns with grepl(make.names(j), ...), a regex in
  # which "." matches the "-" and " " of the raw names; the three columns are M, A, p.
  w <- madata[, paste0(j, c("_M", "_A", "_p_value"))]
  colnames(w) <- c("ma_M", "ma_A", "ma_p_value")
  tmp <- geneinfo[match(rownames(w), geneinfo$reporterId), ]
  tmp <- tmp[match(unique(tmp$geneSymbol), tmp$geneSymbol), ]
  w <- w[match(unique(tmp$reporterId), rownames(w)), ]
  rownames(w) <- geneinfo[match(rownames(w), geneinfo$reporterId), ]$geneSymbol
  w$gene <- rownames(w)
  for (k in 1:3) w[, k] <- suppressWarnings(as.numeric(w[, k]))
  wsig <- sum(abs(w$ma_M) > 1 & w$ma_p_value < 0.05, na.rm = TRUE)
  z <- merge(x, w, by = "gene")
  for (l in 2:5) z[, l] <- as.numeric(z[, l])
  z <- z[rowSums(is.na(z)) == 0, ]
  # As the paper: Spearman on every gene present in both, sentinels included.
  # Also: the same with the +-20 sentinel genes removed, and a Pearson.
  ok <- abs(z[, 2]) < 20
  data.frame(genotype = i, mutant = j, javsma = cor(z[, 2], z[, 3], method = "spearman"),
             javsma_no_sentinel = if (sum(ok) > 30) cor(z[ok, 2], z[ok, 3], method = "spearman") else NA,
             pearson_no_sentinel = if (sum(ok) > 30) cor(z[ok, 2], z[ok, 3]) else NA,
             n_genes = nrow(z), n_sentinel = sum(!ok), MAsig = wsig,
             stringsAsFactors = FALSE)
})
allcors <- do.call(rbind, allcors)
ids <- gsub("bc-", "", allcors$genotype)
allcors$JAnsig <- rowSums(degs[match(ids, degs$genotype), c("upregulated", "downregulated")])
allcors$ncells <- ncells[allcors$genotype]
write.table(allcors, file.path(out_dir, "paper_deleteome_comparison.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

cat("FigS1I as published: Spearman(log NSG_MA, log NSG_SC) =",
    round(cor(log(allcors$MAsig + 1), log(allcors$JAnsig + 1), method = "spearman", use = "complete.obs"), 3),
    " n =", sum(complete.cases(allcors[, c("MAsig", "JAnsig")])), "\n")
cat("unshown javsma: median", round(median(allcors$javsma, na.rm = TRUE), 3),
    " IQR", paste(round(quantile(allcors$javsma, c(.25, .75), na.rm = TRUE), 3), collapse = " .. "),
    " > 0.2:", sum(allcors$javsma > 0.2, na.rm = TRUE), "\n")
cat("NSG_SC vs cells: Spearman", round(cor(allcors$JAnsig, allcors$ncells, method = "spearman", use = "complete.obs"), 3),
    " NSG_MA vs cells:", round(cor(allcors$MAsig, allcors$ncells, method = "spearman", use = "complete.obs"), 3), "\n")

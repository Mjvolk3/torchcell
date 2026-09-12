# experiments/028-knockout-expression/scripts/probe_seurat_split.R
# [[experiments.028-knockout-expression.scripts.probe_seurat_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/probe_seurat_split
#
# What is inside `seus_split.RData` (Nadal-Ribelles 2025, Zenodo 14062629)? Prints the
# object names, classes, assay names and layers, dimensions, and the metadata columns
# with a few example values, so the extraction script can be written against what the
# file actually holds rather than what the README says.
#
#   ~/miniconda3/envs/r-seurat/bin/Rscript experiments/028-knockout-expression/scripts/probe_seurat_split.R
suppressMessages(library(Seurat))
raw <- Sys.getenv("NADAL_RAW", "/scratch/projects/torchcell-scratch/torchcell-raw/nadalRibelles2025")
objs <- load(file.path(raw, "seus_split.RData"))
cat("objects:", paste(objs, collapse = ", "), "\n")
for (nm in objs) {
  x <- get(nm)
  cat("\n==", nm, "class:", paste(class(x), collapse = "/"), "\n")
  if (is.list(x) && !inherits(x, "Seurat")) {
    cat("  list names:", paste(names(x), collapse = ", "), "\n")
    for (k in names(x)) {
      s <- x[[k]]
      cat("  --", k, "class", paste(class(s), collapse = "/"), "\n")
      if (inherits(s, "Seurat")) {
        cat("     assays:", paste(Assays(s), collapse = ", "), " default:", DefaultAssay(s), "\n")
        for (a in Assays(s)) {
          cat("     assay", a, "layers:", paste(Layers(s[[a]]), collapse = ", "), " dim:", paste(dim(s[[a]]), collapse = " x "), "\n")
        }
        md <- s[[]]
        cat("     cells:", ncol(s), " metadata columns:", paste(colnames(md), collapse = ", "), "\n")
        for (cn in colnames(md)) {
          v <- md[[cn]]
          if (is.numeric(v)) cat("       ", cn, ": numeric, range", paste(signif(range(v, na.rm = TRUE), 4), collapse = " .. "), "\n")
          else cat("       ", cn, ": ", length(unique(v)), " unique, e.g. ", paste(head(unique(v), 5), collapse = " | "), "\n")
        }
      }
    }
  } else if (inherits(x, "Seurat")) {
    cat("  assays:", paste(Assays(x), collapse = ", "), " cells:", ncol(x), "\n")
    md <- x[[]]
    cat("  metadata columns:", paste(colnames(md), collapse = ", "), "\n")
  }
}

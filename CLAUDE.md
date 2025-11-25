## Programming Guide

- Do NOT ever use fallback mechanisms unless we clearly tell you to. This means minimize try except blocks, unnecessary conditionals, etc.

## Dendron Paths

We use dendron paths in markdown files to link between project notes. These are very useful for giving additional context.

File of these patters:

[[2025.05.10 - Inspecting Data in GoGraph|dendron://torchcell/torchcell.models.dcell#20250510---inspecting-data-in-gograph]]

Can be found here:

notes/torchcell.models.dcell.md

**The General Pattern**

`notes/` dir is skipped in path description as it is default location

Dendron encode from `torchcell/torchcell.models.dcell` to `notes/torchcell.models.dcell.md`

When I tell you to write some output to a file that is in `notes/` then typially you just need to append or modify, we don't want you messing up dendron frontmatter.

## Saving Images in Python

All images should be saved in `ASSET_IMAGES_DIR`

Do this by using `load_dotenv` and time stamp the images with by using torchcell/timestamp.py

The common patters is `(osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png"))`

## Running Python Files

Don't recommend to run python files. After editing files I will either run them from the terminal or debugger.

## Code Execution

~/miniconda3/envs/torchcell/bin/python script.py

~/miniconda3/envs/torchcell/bin/python -m pytest                                                                                        │
│   tests/torchcell/transforms/test_coo_regression_to_classification.py::TestCOOLabelNormalizationTransform::test_inverse_minmax_coo -xvs

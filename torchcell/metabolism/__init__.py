# torchcell/metabolism/__init__.py
# [[torchcell.metabolism]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/__init__.py
"""Metabolic modeling: yeast-GEM wrapper, flux layer, pathways, media, kinetics.

The package marker exists so coverage.py (``source = ["torchcell"]``) and
``pkgutil.walk_packages`` see these modules; without it the live, tested
``flux_layer`` / ``yeast_GEM`` / ``pathway`` code was invisible to both. Nothing is
re-exported here on purpose: ``yeast_GEM`` imports cobra and downloads on first use, so
callers import the submodule they need.
"""

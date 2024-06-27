import datetime
import os
import sys
import torchcell_sphinx_theme

# Correctly adjust the path to the torchcell module
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../torchcell"))
)
sys.path.append(os.path.join(os.path.dirname(torchcell_sphinx_theme.__file__), "extension"))

import torchcell  # Import torchcell module here
from torchcell import __version__

project = "torchcell"
author = "Your Name"
version = __version__
release = __version__
copyright = f"{datetime.datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "pyg",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "torchcell_sphinx_theme"
html_logo = (
    "https://raw.githubusercontent.com/pyg-team/torchcell_sphinx_theme/"
    "master/torchcell_sphinx_theme/static/img/pyg_logo.png"
)
html_favicon = (
    "https://raw.githubusercontent.com/pyg-team/torchcell_sphinx_theme/"
    "master/torchcell_sphinx_theme/static/img/favicon.png"
)
html_static_path = ["_static"]

add_module_names = False
autodoc_member_order = "bysource"
suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "torch": ("https://pytorch.org/docs/master", None),
}

nbsphinx_thumbnails = {
    "tutorial/create_gnn": "_static/thumbnails/create_gnn.png",
    "tutorial/heterogeneous": "_static/thumbnails/heterogeneous.png",
    "tutorial/create_dataset": "_static/thumbnails/create_dataset.png",
    "tutorial/load_csv": "_static/thumbnails/load_csv.png",
    "tutorial/explain": "_static/thumbnails/explain.png",
    "tutorial/shallow_node_embeddings": "_static/thumbnails/shallow_node_embeddings.png",
    "tutorial/multi_gpu_vanilla": "_static/thumbnails/multi_gpu_vanilla.png",
}


def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {"torchcell": torchcell}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect("source-read", rst_jinja_render)
    app.add_js_file("js/version_alert.js")

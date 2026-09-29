"""Sphinx configuration for the TorchCell documentation (https://mjvolk3.github.io/torchcell/)."""

import inspect
import os
import sys
from typing import Any

import torchcell_sphinx_theme
from sphinx.application import Sphinx

# The repository root, so `import torchcell` resolves to this checkout when the package
# is not installed. CI installs it with `pip install -e .`, which resolves the same way.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(
    os.path.join(os.path.dirname(torchcell_sphinx_theme.__file__), "extension")
)

import torchcell  # noqa: E402  (after the sys.path setup above)
from torchcell import __version__  # noqa: E402

project = "TorchCell"
author = "Michael Volk"
version = __version__
release = __version__
copyright = f"2026, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
    "pyg",
]

# `linkify` is left out: it needs linkify-it-py, which docs/requirements.txt does not
# install.
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "torchcell_sphinx_theme"
html_logo = "_static/torchcell-logo.png"
html_favicon = "_static/torchcell-logo.png"
html_static_path = ["_static"]
html_baseurl = "https://mjvolk3.github.io/torchcell/"

# Copied verbatim into the site root, no theme and no RST wrapper. The docs workflow
# renders the ontology explorer into `_extra/ontology/index.html` just before the
# Sphinx build, which publishes it at https://mjvolk3.github.io/torchcell/ontology/ --
# the URL printed on the ontology figures (see EXPLORE_URL in
# paper/nature-biotech/scripts/generate_ontology_diagram.py).
html_extra_path = ["_extra"]

add_module_names = False
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
    "pydantic": ("https://pydantic.dev/docs/validation/latest", None),
}


def drop_builtin_type_docstring(
    app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    """Blank a module constant's docstring when it is only its builtin type's docstring.

    A module-level ``dict`` or ``list`` has no docstring of its own, so autodoc falls
    back to ``dict.__doc__``, which is not about the constant and is not valid RST.
    """
    type_doc = type(obj).__doc__
    if what != "data" or type(obj).__module__ != "builtins" or type_doc is None:
        return
    if "\n".join(lines).strip() == inspect.cleandoc(type_doc).strip():
        lines[:] = []


def setup(app: Sphinx) -> None:
    """Connect the docstring filter and the Jinja pass over the .rst sources."""
    app.connect("autodoc-process-docstring", drop_builtin_type_docstring)

    # Renders `{% for ... %}` loops in the .rst API pages against the imported package.
    # Markdown pages are skipped: their code samples may contain literal `{{ }}`.
    def rst_jinja_render(app: Sphinx, docname: str, source: list[str]) -> None:
        if not str(app.env.doc2path(docname)).endswith(".rst"):
            return
        rst_context = {"torchcell": torchcell}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect("source-read", rst_jinja_render)

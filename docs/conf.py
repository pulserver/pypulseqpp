"""Configuration for the pypulseqpp documentation."""

from __future__ import annotations

import os

project = "pypulseqpp"
copyright = "2026, pypulseqpp contributors"  # noqa: A001
author = "pypulseqpp contributors"

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "linkify"]
myst_footnote_transition = False

autosummary_generate = True
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_class_signature = "mixed"
autodoc_preserve_defaults = True

napoleon_numpy_docstring = True
napoleon_use_admonition_for_references = True

pygments_style = "sphinx"
highlight_language = "python"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

DOCS_VERSION = os.environ.get("PYPULSEQPP_DOCS_VERSION", "latest")
PAGES_URL = "https://pulserver.github.io/pypulseqpp"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/pulserver/pypulseqpp",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
}
html_baseurl = f"{PAGES_URL}/{DOCS_VERSION}/"
html_title = "pypulseqpp documentation"

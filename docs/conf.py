"""Configuration for the pypulseqpp documentation."""

from __future__ import annotations

import os
import urllib.error
import urllib.request

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

intersphinx_timeout = 5

_INTERSPHINX = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


def _has_inventory(base: str) -> bool:
    """Whether this project's object inventory can be fetched right now.

    Sphinx reports an inventory it cannot reach as a warning carrying no type,
    which `suppress_warnings` therefore cannot name and which the build's `-W`
    turns into a failure. Leaving such a project out of the mapping costs the
    cross-references into it, which render as their own text, and keeps an
    outage elsewhere from failing this build.
    """
    url = base.rstrip("/") + "/objects.inv"
    try:
        # The body is never read: this asks whether the inventory is served,
        # and intersphinx fetches it in full when it is.
        with urllib.request.urlopen(url, timeout=intersphinx_timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


intersphinx_mapping = {}
for _project, _entry in _INTERSPHINX.items():
    if _has_inventory(_entry[0]):
        intersphinx_mapping[_project] = _entry
    else:
        print(f"conf.py: {_entry[0]} is unreachable; building without its links")

DOCS_VERSION = os.environ.get("PYPULSEQPP_DOCS_VERSION", "latest")
PAGES_URL = "https://pulserver.github.io/pypulseqpp"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/pulserver/pypulseqpp",
    # Where the pages live in the repository: the edit button links to the
    # source file under it, and defaults to the repository root without this.
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
}
html_baseurl = f"{PAGES_URL}/{DOCS_VERSION}/"
html_title = "pypulseqpp documentation"

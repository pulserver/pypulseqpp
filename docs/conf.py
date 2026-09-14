"""Configuration for the pypulseqpp documentation."""

from __future__ import annotations

import logging
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

intersphinx_timeout = 10

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


class _InventoryOutageFilter(logging.Filter):
    """Drop the warning intersphinx logs when it cannot fetch an inventory.

    The message carries no warning type, so `suppress_warnings` has no name to
    match it by, and the build runs under `-W`: an outage at somebody else's
    documentation host would fail this build. Cross-references into a project
    whose inventory is missing render as their own text instead.

    Deciding reachability here rather than in the mapping matters: a check of
    our own is a second fetch with its own headers and timeout, and a project
    it judges unreachable but intersphinx could have read loses every link
    into it silently.
    """

    _MESSAGE = "failed to reach any of the inventories"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._MESSAGE not in record.getMessage()


def setup(_app):
    """Install the filter ahead of Sphinx's own, which count the warning."""
    handlers = logging.getLogger("sphinx").handlers
    if not handlers:
        print("conf.py: no Sphinx log handler; an inventory outage will fail the build")
    for handler in handlers:
        handler.filters.insert(0, _InventoryOutageFilter())


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

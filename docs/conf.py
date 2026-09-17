"""Configuration for the pypulseqpp documentation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder

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
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

templates_path = ["_templates"]
# A gallery header is Markdown pulled into the generated index.rst by an
# include, so it travels into the output directory beside it and must not also
# be built as a page of its own.
exclude_patterns = [
    "build",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/_gallery_header.md",
]

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
# An Attributes section renders as a field list, as Parameters does, rather
# than as one attribute directive per entry.
napoleon_custom_sections = [("Attributes", "params_style")]

pygments_style = "sphinx"
highlight_language = "python"

intersphinx_timeout = 10

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

#: The gallery's sections, in the order a reader should meet them.
GALLERY_SECTIONS = [
    "../gallery/01-cartesian",
    "../gallery/02-non-cartesian",
    "../gallery/03-advanced-design",
]

sphinx_gallery_conf = {
    "doc_module": "pypulseqpp",
    "backreferences_dir": "generated/gallery_backreferences",
    "reference_url": {"pypulseqpp": None},
    "examples_dirs": ["../gallery"],
    "gallery_dirs": ["generated/gallery"],
    # Every script is executed: the sections are ordered by the list above and
    # the scripts within a section by the numeric prefix of their file names.
    "filename_pattern": r".*\.py",
    "nested_sections": True,
    "subsection_order": ExplicitOrder(GALLERY_SECTIONS),
    "within_subsection_order": "FileNameSortKey",
    # Left off deliberately: it would strip the ignore flags before the page is
    # written, and _hide_ignored_code_from_the_page_only needs them there.
    "remove_config_comments": False,
    # Two files have to travel into the output directory: the MyST section
    # headers, which a generated index.rst pulls in by an include, and our own
    # root index.rst. Matching `index.rst` here is also what makes
    # sphinx-gallery use ours: `_get_gallery_header` returns None for a
    # directory holding an `index.rst` that this pattern matches, and the root
    # index it would otherwise write is an orphan whose toctree flattens every
    # example into the top level of the sidebar.
    "copyfile_regex": r"(.*\.md|index\.rst)",
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


def _public_bases(_app, _name, _obj, _options, bases):
    """List a private base as the nearest public class it is built on.

    A private class has no page to link to, and a public class built on one
    carries its documentation already.
    """
    bases[:] = [
        next(klass for klass in base.__mro__ if not klass.__name__.startswith("_"))
        if isinstance(base, type)
        else base
        for base in bases
    ]


def _hide_ignored_code_from_the_page_only() -> None:
    """Strip an example's hidden blocks from the page and from nothing else.

    sphinx-gallery removes the regions between ``sphinx_gallery_start_ignore``
    and ``sphinx_gallery_end_ignore`` once, before it writes either the page or
    the downloadable notebook, so the notebook is missing whatever the page
    hides and fails on the first cell that needed it. Removing them as the page
    is written instead leaves the downloadable script and notebook executable.

    A cell hidden in full renders as nothing rather than as an empty
    ``code-block`` directive. Its output is emitted separately and is kept
    either way.
    """
    from sphinx_gallery import gen_rst, py_source_parser

    strip = py_source_parser.remove_ignore_blocks

    def keep(code):
        strip(code)  # for its check that every flag has its partner
        return code

    py_source_parser.remove_ignore_blocks = keep

    original = gen_rst.codestr2rst

    def codestr2rst(code, *args, **kwargs):
        shown = strip(code)
        return original(shown, *args, **kwargs) if shown.strip() else ""

    gen_rst.codestr2rst = codestr2rst

    write_notebook = gen_rst.jupyter_notebook

    def jupyter_notebook(script_blocks, *args, **kwargs):
        """The notebook keeps the code, without the flags that hid it."""
        return write_notebook(
            [
                block._replace(content=_unflagged(block.content))
                for block in script_blocks
            ],
            *args,
            **kwargs,
        )

    gen_rst.jupyter_notebook = jupyter_notebook


def _unflagged(content: str) -> str:
    """``content`` without the comment lines that mark a hidden region."""
    return "\n".join(
        line
        for line in content.splitlines()
        if line.strip()
        not in ("# sphinx_gallery_start_ignore", "# sphinx_gallery_end_ignore")
    )


def _draw_explanation_figures(app) -> None:
    """Render the explanation pages' figures with the pypulseqpp being built.

    They are designed and analysed at build time rather than drawn once and
    committed, so a figure cannot outlive the behaviour it reports.
    """
    import sys

    sys.path.insert(0, str(Path(app.srcdir)))
    from explanation_figures import render

    render(Path(app.srcdir, "generated", "figures"))


def setup(app):
    """Install the filter ahead of Sphinx's own, which count the warning."""
    _hide_ignored_code_from_the_page_only()
    app.connect("autodoc-process-bases", _public_bases)
    app.connect("builder-inited", _draw_explanation_figures)
    handlers = logging.getLogger("sphinx").handlers
    if not handlers:
        print("conf.py: no Sphinx log handler; an inventory outage will fail the build")
    for handler in handlers:
        handler.filters.insert(0, _InventoryOutageFilter())


DOCS_VERSION = os.environ.get("PYPULSEQPP_DOCS_VERSION", "latest")
PAGES_URL = "https://pulserver.github.io/pypulseqpp"

#: Which published version this build is: ``latest`` for main, the tag for a
#: release. The switcher marks it and warns on a page older than the newest
#: release; ``DOCS_VERSION`` is the canonical name instead, ``stable`` for a
#: release, which is what the page links point at.
DOCS_RELEASE = os.environ.get("PYPULSEQPP_DOCS_RELEASE", "latest")

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
    # The list every published version is in, written beside the versions by
    # scripts/publish_docs.py. The page fetches it when it loads, so a build
    # served from anywhere else leaves the switcher out. The theme's check of
    # the list at build time is off: the list exists only once a version has
    # been published.
    "switcher": {
        "json_url": f"{PAGES_URL}/versions.json",
        "version_match": DOCS_RELEASE,
    },
    "check_switcher": False,
    "show_version_warning_banner": True,
    # The sidebar carries the hierarchy, not every leaf: sections and the
    # pages under them, and no deeper. Individual examples are reached from
    # the gallery's category pages, and individual functions, classes and
    # methods from the tables on the API pages and from each page's own
    # contents list.
    "max_navbar_depth": 2,
    "show_navbar_depth": 1,
}

#: The theme's own sidebar, with the version switcher under the title.
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "version-switcher.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}
html_baseurl = f"{PAGES_URL}/{DOCS_VERSION}/"
html_title = "pypulseqpp documentation"

"""Open each gallery example in Google Colab, from a notebook that installs what it needs.

Colab opens a notebook from a GitHub repository, so the notebooks it opens are
copies published with the site on the ``gh-pages`` branch, under
``<version>/_colab/``.  Each copy is the notebook sphinx-gallery writes for the
example, with two cells in front of it: a note, and a ``%pip install`` of
pypulseqpp and the packages the example imports.  The notebooks the example
pages offer for download are left as sphinx-gallery writes them.

``badge`` is the reStructuredText each example page carries under its title,
and ``write`` puts the copies into the built site.
"""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path

#: The repository whose gh-pages branch holds the published site.
REPOSITORY = "pulserver/pypulseqpp"

#: What every example installs beside pypulseqpp, and what a section adds.
PACKAGES = ["matplotlib"]
SECTION_PACKAGES = {"13-fast-spin-echo": ["torchsim"]}

#: The gallery's output directory under the documentation sources.
GALLERY = "generated/gallery"


def requirement(release: str) -> str:
    """pypulseqpp as the notebook installs it: the release a page documents, or the newest.

    The ``plot`` extra is the viewer :meth:`pypulseqpp.Sequence.plot` draws in.
    """
    name = "pypulseqpp[plot]"
    return name if release == "latest" else f"{name}=={release.removeprefix('v')}"


def colab_url(relative: str, release: str) -> str:
    """The Colab URL of the notebook for ``relative``, ``<section>/<name>`` without a suffix."""
    return (
        f"https://colab.research.google.com/github/{REPOSITORY}/blob/gh-pages/"
        f"{release}/_colab/{relative}.ipynb"
    )


def setup_cells(section: str, release: str) -> list[dict]:
    """The note and the install cell a Colab copy starts with."""
    packages = [requirement(release), *PACKAGES, *SECTION_PACKAGES.get(section, [])]
    text = (
        "Setup for Colab: installs pypulseqpp and the packages this example uses. "
        "It is not part of the example."
    )
    if release == "latest":
        text += (
            " This page documents the development version, and the newest pypulseqpp "
            "release installed here can lack what the example uses."
        )
    note = {"cell_type": "markdown", "metadata": {}, "source": [text]}
    install = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["%pip install --quiet " + " ".join(shlex.quote(p) for p in packages)],
    }
    return [note, install]


def colab_notebook(notebook: dict, section: str, release: str) -> dict:
    """``notebook`` with the setup cells in front."""
    out = copy.deepcopy(notebook)
    out["cells"] = setup_cells(section, release) + out["cells"]
    return out


def badge(docname: str, release: str) -> str | None:
    """The badge an example page carries under its title; ``None`` for a section page."""
    if not docname.startswith(GALLERY + "/"):
        return None
    parts = docname[len(GALLERY) + 1 :].split("/")
    if len(parts) != 2 or parts[1] == "index":
        return None
    url = colab_url(f"{parts[0]}/{parts[1]}", release)
    return (
        ".. only:: html\n\n"
        "   .. image:: https://colab.research.google.com/assets/colab-badge.svg\n"
        f"      :target: {url}\n"
        "      :alt: Open in Colab\n"
        "      :class: colab-badge\n"
    )


def with_badge(text: str, docname: str, release: str) -> str:
    """The generated page of an example with the badge inserted under its title."""
    block = badge(docname, release)
    if block is None:
        return text
    lines = text.split("\n")
    for i in range(len(lines) - 2):
        rule = lines[i].strip()
        if rule and set(rule) == {"="} and lines[i + 2].strip() == rule:
            lines[i + 3 : i + 3] = ["", block]
            return "\n".join(lines)
    return text


def write(source: Path, site: Path, release: str) -> int:
    """Write the Colab copy of every gallery notebook under ``site/_colab``; return the count."""
    count = 0
    for notebook in sorted((source / GALLERY).glob("*/*.ipynb")):
        section = notebook.parent.name
        target = site / "_colab" / section / notebook.name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = json.loads(notebook.read_text(encoding="utf-8"))
        target.write_text(
            json.dumps(colab_notebook(content, section, release), indent=1) + "\n",
            encoding="utf-8",
        )
        count += 1
    return count

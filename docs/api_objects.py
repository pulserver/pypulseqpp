"""Collect the API pages' autosummary blocks into one page outside the navigation.

The API pages present their objects as tables. The stub page for each object is
generated from this module's output instead of from the tables themselves, so
that the stubs stay out of the toctree the sidebar is built from: the Examples
section is three levels deep, and a navigation depth that shows its sequence
families would otherwise show every method and attribute stub as well.

The object lists are read from the API pages, which remain the only place they
are written.
"""

from __future__ import annotations

import re
from pathlib import Path

#: An ``autosummary`` block inside an ``eval-rst`` fence, with its options and
#: its entries.
_BLOCK = re.compile(r"^\.\. autosummary::\n((?:[ \t]+.*\n|\n)*)", re.M)

#: The module an API page documents, declared once near its top.
_CURRENTMODULE = re.compile(r"^\.\. currentmodule:: (\S+)$", re.M)


def _entries(body: str) -> tuple[list[str], list[str]]:
    """The options and the object names of one ``autosummary`` block."""
    options, names = [], []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        (options if stripped.startswith(":") else names).append(stripped)
    return options, names


def collect(pages: Path) -> list[tuple[str, list[str], list[str]]]:
    """Every API page's blocks, as ``(module, options, names)``."""
    blocks = []
    for page in sorted(pages.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        module = _CURRENTMODULE.search(text)
        for match in _BLOCK.finditer(text):
            options, names = _entries(match.group(1))
            if names:
                blocks.append((module.group(1) if module else "", options, names))
    return blocks


def render(blocks: list[tuple[str, list[str], list[str]]]) -> str:
    """The holder page: every block again, this time writing its stubs."""
    lines = [
        ":orphan:",
        "",
        "API object index",
        "================",
        "",
        "The stub page of every documented object. The API pages carry the same",
        "object lists as tables, each entry linking the stub this page writes;",
        "writing them here keeps them out of the toctree the sidebar is built",
        "from.",
        "",
    ]
    module = None
    for owner, options, names in blocks:
        if owner and owner != module:
            module = owner
            lines += [f".. currentmodule:: {module}", ""]
        kept = [option for option in options if not option.startswith(":toctree:")]
        lines += [".. autosummary::", "   :toctree: generated"]
        lines += [f"   {option}" for option in kept]
        lines += [""] + [f"   {name}" for name in names] + [""]
    return "\n".join(lines) + "\n"


def write(into: str | Path) -> int:
    """Write the holder page under ``into``; return the number of objects.

    The page sits at the top of the source directory so that its ``:toctree:``
    resolves to ``generated/``, where the stubs the API pages link already are.
    """
    root = Path(into)
    blocks = collect(root / "api")
    (root / "generated").mkdir(parents=True, exist_ok=True)
    (root / "api_objects.rst").write_text(render(blocks), encoding="utf-8")
    return sum(len(names) for _, _, names in blocks)

"""Load required compiled kernels and report missing or incompatible binaries."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = ["require"]

_MODULE = "pypulseqpp._ext"


def _installed_tags() -> list[str]:
    """Return the ABI tags of the extension builds present beside the package."""
    suffix = ".pyd" if sys.platform == "win32" else ".so"
    package = import_module("pypulseqpp")
    tags = set()
    for directory in getattr(package, "__path__", ()):
        for path in Path(directory).glob(f"_ext.*{suffix}"):
            parts = path.name.split(".")
            if len(parts) >= 3:
                tags.add(parts[1])
    return sorted(tags)


def require(attribute: str | None = None) -> Any:
    """Load the bundled accelerator, or raise naming the mismatch.

    Parameters
    ----------
    attribute
        Symbol the caller needs. Checked here, so a binary that predates the
        symbol fails at the load rather than at the call.

    Returns
    -------
    module or object
        The extension module, or the named attribute when one is given.

    Raises
    ------
    ImportError
        The accelerator is absent, was built for another interpreter, or does
        not provide ``attribute``.
    """
    try:
        module = import_module(_MODULE)
    except ImportError as error:
        installed = _installed_tags()
        found = ", ".join(installed) if installed else "none"
        raise ImportError(
            f"the bundled accelerator did not load. Running "
            f"{sys.implementation.cache_tag}, builds present: {found}. "
            f"This package ships the kernels for every interpreter it "
            f"supports, so install the distribution matching this one."
        ) from error
    if attribute is None:
        return module
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ImportError(
            f"the bundled accelerator does not provide {attribute!r}; "
            f"the installed binary predates it."
        ) from error

"""Execute doctests from shipped modules, excluding merely importable workspace files."""

import doctest
import importlib
import pkgutil
from pathlib import Path

import pytest

import pypulseqpp as package

#: Where the package's own code lives. An editable install puts every prefix
#: of a mapped path on `package.__path__`, so the repo root is on it and
#: anything importable beside `pyproject.toml` walks as `pypulseqpp.<name>`.
#: What is collected is what the package ships, wherever it is written.
_OURS = (
    Path(package.__file__).parent.resolve(),
    # The zoo is written outside the package and mapped in, so its scripts
    # are shipped code whose files are not under the package directory.
    *(
        Path(entry).resolve()
        for entry in importlib.import_module("pypulseqpp.sequences.sequence").__path__
    ),
)


def _shipped(info) -> bool:
    """Check file ownership before import; editable paths can expose executable setup.py."""
    spec = info.module_finder.find_spec(info.name)
    origin = getattr(spec, "origin", None)
    if origin is None:
        return False
    return any(root in Path(origin).resolve().parents for root in _OURS)


def _modules():
    yield package
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if _shipped(info):
            yield importlib.import_module(info.name)


@pytest.mark.parametrize("module", list(_modules()), ids=lambda module: module.__name__)
def test_every_docstring_example_runs(module, capsys):
    results = doctest.testmod(
        module, verbose=False, report=False, optionflags=doctest.ELLIPSIS
    )
    captured = capsys.readouterr()
    assert results.failed == 0, captured.out

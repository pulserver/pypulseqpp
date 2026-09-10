"""Shared reference-toolbox and native-extension sequence fixtures."""

import pytest

try:
    import extended
    import reference
except ImportError:  # the reference toolbox is not installed; see reference.py
    extended = reference = None


def _names(module):
    """The sequence names a module offers, or one that skips without it."""
    return sorted(module.ZOO) if module else ["the reference toolbox is absent"]


@pytest.fixture(params=_names(reference), ids=lambda name: name)
def reference_name(request):
    """Each reference sequence in turn, by name."""
    if reference is None:
        pytest.skip("needs pypulseq-matlab-like; see tests/reference.py")
    return request.param


@pytest.fixture
def build_reference(reference_name):
    """Build the current reference sequence, fresh each time it is called."""
    return reference.ZOO[reference_name]


@pytest.fixture(params=_names(extended), ids=lambda name: name)
def extended_name(request):
    """Each sequence using a 1.5.1 event kind in turn, by name."""
    if extended is None:
        pytest.skip("needs pypulseq-matlab-like; see tests/reference.py")
    return request.param


@pytest.fixture
def build_extended(extended_name):
    """Build the current extended sequence, fresh each time it is called."""
    return extended.ZOO[extended_name]

"""Shared fixtures.

The reference sequences are built by the toolbox pypulseqpp replaces, so every
test that uses one is comparing against the implementation of record rather
than against a value someone wrote down.
"""

import pytest
import reference


@pytest.fixture(params=sorted(reference.ZOO), ids=lambda name: name)
def reference_name(request):
    """Each reference sequence in turn, by name."""
    return request.param


@pytest.fixture
def build_reference(reference_name):
    """Build the current reference sequence, fresh each time it is called."""
    return reference.ZOO[reference_name]

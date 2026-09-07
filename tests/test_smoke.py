"""The package imports and reports a version."""

import pypulseqpp


def test_the_package_reports_a_version():
    assert isinstance(pypulseqpp.__version__, str)
    assert pypulseqpp.__version__

"""The shape codec encodes what PyPulseq's encodes.

A `[SHAPES]` entry is run-length encoded on its derivative, so a gradient that
ramps and holds costs a handful of numbers rather than its whole raster. The
encoding is part of the file format, which is why the test of record is not a
property but a comparison: our encoder against the reference one, sample for
sample.
"""

import numpy as np
import pytest
from pypulseq.compress_shape import compress_shape as upstream_compress

from pypulseqpp import _ext

WAVEFORMS = {
    "constant": np.ones(100),
    "linear_ramp": np.linspace(0, 1, 100),
    "trapezoid": np.concatenate(
        [np.linspace(0, 1, 20), np.ones(60), np.linspace(1, 0, 20)]
    ),
    "sinusoid": np.sin(np.linspace(0, 4 * np.pi, 300)),
    "sinc": np.sinc(np.linspace(-4, 4, 512)),
    "noise": np.random.default_rng(42).standard_normal(100),
    "four_samples": np.array([1.0, 2.0, 3.0, 4.0]),
    "one_sample": np.array([5.0]),
}


@pytest.fixture(params=sorted(WAVEFORMS), ids=lambda name: name)
def waveform(request):
    return WAVEFORMS[request.param]


def test_the_encoding_matches_the_reference_encoder(waveform):
    expected = np.asarray(upstream_compress(waveform).data, dtype=float)
    np.testing.assert_array_equal(_ext.compress_shape(waveform), expected)


def test_decoding_recovers_the_waveform(waveform):
    encoded = _ext.compress_shape(waveform)
    decoded = _ext.decompress_shape(encoded, len(waveform))
    np.testing.assert_allclose(decoded, waveform, atol=1e-6)


def test_decoding_yields_the_sample_count_it_was_asked_for(waveform):
    encoded = _ext.compress_shape(waveform)
    assert len(_ext.decompress_shape(encoded, len(waveform))) == len(waveform)


def test_a_constant_waveform_costs_a_handful_of_numbers():
    assert len(_ext.compress_shape(np.ones(1000))) == 4


def test_an_incompressible_waveform_is_kept_as_it_stands():
    noise = np.random.default_rng(7).standard_normal(200)
    np.testing.assert_array_equal(_ext.compress_shape(noise), noise)


def test_four_samples_or_fewer_are_never_encoded():
    for count in (1, 2, 3, 4):
        samples = np.ones(count)
        np.testing.assert_array_equal(_ext.compress_shape(samples), samples)


def test_an_entry_that_was_never_encoded_decodes_to_itself():
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(_ext.decompress_shape(samples, len(samples)), samples)

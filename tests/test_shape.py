"""Shape-codec parity with the reference implementation."""

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)
from pypulseq_matlab_like.compress_shape import compress_shape as reference_compress

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
    expected = np.asarray(reference_compress(waveform).data, dtype=float)
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


def test_a_shape_starts_with_no_role():
    sequence = _ext.Sequence()
    sequence.register_shape(4, np.array([0.0, 1.0, 1.0, 0.0]))
    assert list(sequence.shape_roles()) == [int(_ext.ShapeRole.NONE)]


def test_registering_an_rf_event_marks_its_magnitude_and_phase():
    sequence = _ext.Sequence()
    magnitude = sequence.register_shape(4, np.array([0.0, 1.0, 1.0, 0.0]))
    phase = sequence.register_shape(4, np.array([0.0, 0.5, 0.5, 0.0]))
    row = np.zeros(10)
    row[0], row[1], row[2] = 100.0, magnitude, phase

    sequence.register_rf(row, "e")

    assert list(sequence.shapes_with_role(int(_ext.ShapeRole.RF_MAGNITUDE))) == [
        magnitude
    ]
    assert list(sequence.shapes_with_role(int(_ext.ShapeRole.RF_PHASE))) == [phase]


def test_registering_an_arbitrary_gradient_marks_its_waveform_and_times():
    sequence = _ext.Sequence()
    waveform = sequence.register_shape(4, np.array([0.0, 1.0, 1.0, 0.0]))
    times = sequence.register_shape(4, np.array([0.0, 1.0, 2.0, 3.0]))
    row = np.zeros(6)
    row[0], row[3], row[4] = 1000.0, waveform, times

    sequence.register_arbitrary(row)

    assert list(sequence.shapes_with_role(int(_ext.ShapeRole.GRADIENT))) == [waveform]
    assert list(sequence.shapes_with_role(int(_ext.ShapeRole.GRADIENT_TIME))) == [times]
    assert list(sequence.shapes_with_role(int(_ext.ShapeRole.TIME))) == [times]


def test_registering_an_adc_marks_its_phase_modulation():
    sequence = _ext.Sequence()
    modulation = sequence.register_shape(4, np.array([0.0, 1.0, 2.0, 3.0]))
    row = np.zeros(8)
    row[0], row[7] = 128.0, modulation

    sequence.register_adc(row)

    assert list(sequence.shapes_with_role(int(_ext.ShapeRole.ADC_PHASE))) == [
        modulation
    ]


def test_a_shape_played_two_ways_carries_both_roles():
    sequence = _ext.Sequence()
    shared = sequence.register_shape(4, np.array([0.0, 1.0, 1.0, 0.0]))
    rf = np.zeros(10)
    rf[0], rf[1] = 100.0, shared
    gradient = np.zeros(6)
    gradient[0], gradient[3] = 1000.0, shared

    sequence.register_rf(rf, "e")
    sequence.register_arbitrary(gradient)

    both = int(_ext.ShapeRole.RF_MAGNITUDE) | int(_ext.ShapeRole.GRADIENT)
    assert sequence.shape_roles()[shared - 1] == both


def test_collapsing_duplicates_merges_the_roles_of_the_shapes_it_merges():
    sequence = _ext.Sequence()
    samples = np.array([0.0, 1.0, 1.0, 0.0])
    as_rf = sequence.register_shape(4, samples)
    as_gradient = sequence.register_shape(4, samples)
    rf = np.zeros(10)
    rf[0], rf[1] = 100.0, as_rf
    gradient = np.zeros(6)
    gradient[0], gradient[3] = 1000.0, as_gradient
    sequence.register_rf(rf, "e")
    sequence.register_arbitrary(gradient)
    sequence.add_block(1, 1, 0, 0, 0, 0, 4e-5)

    sequence.remove_duplicates()

    roles = sequence.shape_roles()
    assert len(roles) == 1
    assert roles[0] == int(_ext.ShapeRole.RF_MAGNITUDE) | int(_ext.ShapeRole.GRADIENT)

"""3D Cartesian example sequences: view sampling, calibration region and wave-CAIPI."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp

#: A prescription small enough to build in a moment, per sequence.
SMALL = {
    "gre3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8},
    "gre_multiecho3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8, "n_echoes": 2},
    "se3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8, "tr": None},
}

APPS = {
    "gre3D_sequence": "Gre3DApp",
    "gre_multiecho3D_sequence": "GreMultiecho3DApp",
    "se3D_sequence": "Se3DApp",
}


def app(name, **kwargs):
    module = importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")
    cls = getattr(module, APPS[name])
    return cls(pp.Opts(), **{**SMALL[name], "n_dummy": 0, **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def acquisitions(seq):
    """The blocks that acquire, in play order."""
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    return [block for block in blocks if block.adc is not None]


def excitations(seq):
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    return sum(
        block.rf is not None and block.rf.use != "refocusing" for block in blocks
    )


def inside(view, centre, extent):
    """Whether ``view`` lies in the ellipse of diameters ``extent`` about ``centre``."""
    return (
        sum(
            ((v - c) / (d / 2)) ** 2
            for v, c, d in zip(view, centre, extent, strict=True)
        )
        <= 1
    )


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("n_dummy", [0, 3])
def test_the_scan_opens_with_the_dummies_asked_for(name, n_dummy):
    built = app(name, n_dummy=n_dummy)
    seq = built.design()

    assert excitations(seq) == n_dummy + len(built.views)
    assert len(acquisitions(seq)) == len(built.views) * getattr(built, "n_echoes", 1)


@pytest.mark.parametrize("name", SMALL)
def test_elliptical_sampling_keeps_the_views_inside_the_inscribed_ellipse(name):
    full = app(name, ry=2, rz=2, n_acs_y=4, n_acs_z=2, elliptical_sampling=False)
    cropped = app(name, ry=2, rz=2, n_acs_y=4, n_acs_z=2)
    n_y, n_z = full.matrix[1:]
    centre, grid = (n_y // 2, n_z // 2), (n_y, n_z)

    kept = [view for view in full.views if inside(view, centre, grid)]
    assert cropped.calibration == full.calibration
    assert cropped.views[: len(cropped.calibration)] == sorted(cropped.calibration)
    assert set(cropped.views) == set(kept) | cropped.calibration
    assert len(cropped.views) < len(full.views)
    assert centre in cropped.views


@pytest.mark.parametrize("name", SMALL)
def test_an_elliptical_calibration_region_is_inscribed_in_the_rectangle(name):
    rectangle = app(name, ry=2, rz=2, n_acs_y=8, n_acs_z=6)
    ellipse = app(name, ry=2, rz=2, n_acs_y=8, n_acs_z=6, elliptical_acs=True)
    n_y, n_z = rectangle.matrix[1:]
    centre = (n_y // 2, n_z // 2)

    assert len(rectangle.calibration) == 8 * 6
    assert ellipse.calibration == {
        view for view in rectangle.calibration if inside(view, centre, (8, 6))
    }
    assert (n_y // 2 - 4, n_z // 2 - 3) not in ellipse.calibration
    assert centre in ellipse.calibration
    assert ellipse.views[: len(ellipse.calibration)] == sorted(ellipse.calibration)


@pytest.mark.parametrize("name", SMALL)
def test_a_fully_sampled_scan_has_no_calibration_region_whatever_its_shape(name):
    assert app(name, elliptical_acs=True, n_acs_y=4, n_acs_z=2).calibration == set()


@pytest.mark.parametrize("name", SMALL)
def test_the_wave_free_reference_leads_and_is_marked_ref(name):
    built = app(
        name,
        wave="both",
        wave_cycles=2,
        wave_amplitude=8e-3,
        ry=2,
        n_acs_y=4,
        n_acs_z=2,
    )
    seq = built.design()
    lin, par, ref, ima, seg = adc_labels(seq, "LIN", "PAR", "REF", "IMA", "SEG")
    echoes = getattr(built, "n_echoes", 1)
    views = [v for v in built.reference + built.views for _ in range(echoes)]
    n_reference = len(built.reference) * echoes

    assert seq.check_timing()[0]
    assert built.reference == sorted(built.calibration)
    assert list(zip(lin, par, strict=True)) == views
    assert list(ref) == [1] * n_reference + [0] * (len(views) - n_reference)
    assert list(seg) == [0] * n_reference + [1] * (len(views) - n_reference)
    assert set(ima) == {0}


@pytest.mark.parametrize("name", SMALL)
def test_the_reference_is_played_with_the_corkscrew_scaled_away(name):
    built = app(
        name,
        wave="phase",
        wave_cycles=2,
        wave_amplitude=8e-3,
        ry=2,
        n_acs_y=4,
        n_acs_z=2,
    )
    reads = acquisitions(built.design())
    n_reference = len(built.reference) * getattr(built, "n_echoes", 1)

    assert n_reference > 0
    assert all(b.gy.amplitude == 0 for b in reads[:n_reference])
    assert all(b.gy.amplitude != 0 for b in reads[n_reference:])


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize(
    "wave",
    [{}, {"wave_amplitude": 8e-3, "wave_cycles": 0}],
    ids=["default", "zero cycles"],
)
def test_without_wave_there_is_no_reference_and_no_wave_gradient(name, wave):
    built = app(name, ry=2, n_acs_y=4, n_acs_z=2, **wave)
    seq = built.design()
    (ref,) = adc_labels(seq, "REF")

    assert built.reference == []
    assert set(ref) == {0}
    assert all(b.gy is None and b.gz is None for b in acquisitions(seq))


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize(
    "prescription",
    [
        {},
        {"wave": "both", "wave_cycles": 2, "wave_amplitude": 8e-3},
        {"elliptical_sampling": False},
    ],
    ids=["plain", "wave", "whole grid"],
)
def test_a_3d_cartesian_scan_repeats_from_its_first_block(name, prescription):
    seq = app(name, n_dummy=2, ry=2, n_acs_y=4, n_acs_z=2, **prescription).design()

    assert seq.repetition()[1] == 1


@pytest.mark.parametrize("name", SMALL)
def test_a_wave_amplitude_beyond_the_system_limits_resolves_to_the_one_built(name):
    wave = {"wave": "both", "wave_cycles": 8, "ry": 2, "n_acs_y": 4, "n_acs_z": 2}
    capped = app(name, wave_amplitude=0.2, **wave)
    amplitude = capped.resolved["wave_amplitude"]
    again = app(name, wave_amplitude=amplitude, **wave)

    assert 0.0 < amplitude < 0.2
    assert again.resolved["wave_amplitude"] == pytest.approx(amplitude)
    assert np.asarray(again.ro.gy_wave.waveform) == pytest.approx(
        np.asarray(capped.ro.gy_wave.waveform)
    )


@pytest.mark.parametrize("name", SMALL)
def test_a_wave_that_is_not_played_resolves_to_zero_amplitude(name):
    built = app(name, wave_amplitude=8e-3, wave_cycles=0)

    assert not built.resolved["wave_amplitude"]


@pytest.mark.parametrize("name", SMALL)
def test_an_unknown_wave_mode_is_refused(name):
    with pytest.raises(ValueError, match="wave mode"):
        app(name, wave="corkscrew")

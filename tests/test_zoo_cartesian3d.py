"""3D Cartesian example sequences: view sampling, calibration region and wave-CAIPI."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences

#: A prescription small enough to build in a moment, per sequence.
SMALL = {
    "gre3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8},
    "gre_multiecho3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8, "n_echoes": 2},
    "se3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8, "tr": None},
}


def built(name, **kwargs):
    """The sequence ``name`` designs, without dummies unless asked for."""
    return getattr(sequences, name).main(**{**SMALL[name], "n_dummy": 0, **kwargs})


def sampled(name, **kwargs):
    """The calibration views and the others the sampling helper lists for ``name``."""
    protocol = {
        **{
            parameter: entry.default
            for parameter, entry in sequences.parameters(
                getattr(sequences, name).main
            ).items()
        },
        **SMALL[name],
        **kwargs,
    }
    return pp.make_cartesian_plane_sampling(
        (protocol["n_y"], protocol["n_z"]),
        (protocol["ry"], protocol["rz"]),
        (protocol["n_acs_y"], protocol["n_acs_z"]),
        caipi_shift=protocol["caipi_shift"],
        partial_fourier=(protocol["partial_fourier_y"], protocol["partial_fourier_z"]),
        elliptical=protocol["elliptical_sampling"],
        elliptical_acs=protocol["elliptical_acs"],
    )


def echoes(name):
    """Echoes read per excitation."""
    return SMALL[name].get("n_echoes", 1)


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


def played(seq, name):
    """The ``(line, partition)`` of each repetition that acquires, in play order, and its ``IMA`` mark."""
    lin, par, ima = adc_labels(seq, "LIN", "PAR", "IMA")
    step = echoes(name)
    views = list(zip(lin[::step].tolist(), par[::step].tolist(), strict=True))
    return views, ima[::step].tolist()


def calibration_of(seq, name):
    """The views played, in order, and the set of those marked as calibration."""
    views, marked = played(seq, name)
    return views, {view for view, mark in zip(views, marked, strict=True) if mark}


def wave_waveform(seq):
    """The wave gradient on the phase encode at the last acquisition (Hz/m)."""
    return np.asarray(acquisitions(seq)[-1].gy.waveform)


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
    seq = built(name, n_dummy=n_dummy)
    calibrating, imaging = sampled(name)
    views = len(calibrating) + len(imaging)

    assert excitations(seq) == n_dummy + views
    assert len(acquisitions(seq)) == views * echoes(name)


@pytest.mark.parametrize("name", SMALL)
def test_elliptical_sampling_keeps_the_views_inside_the_inscribed_ellipse(name):
    acceleration = {"ry": 2, "rz": 2, "n_acs_y": 4, "n_acs_z": 2}
    full, full_calibration = calibration_of(
        built(name, elliptical_sampling=False, **acceleration), name
    )
    cropped, cropped_calibration = calibration_of(built(name, **acceleration), name)
    n_y, n_z = SMALL[name]["n_y"], SMALL[name]["n_z"]
    centre, grid = (n_y // 2, n_z // 2), (n_y, n_z)

    kept = [view for view in full if inside(view, centre, grid)]
    assert cropped_calibration == full_calibration
    assert cropped[: len(cropped_calibration)] == sorted(cropped_calibration)
    assert set(cropped) == set(kept) | cropped_calibration
    assert len(cropped) < len(full)
    assert centre in cropped


@pytest.mark.parametrize("name", SMALL)
def test_an_elliptical_calibration_region_is_inscribed_in_the_rectangle(name):
    region = {"ry": 2, "rz": 2, "n_acs_y": 8, "n_acs_z": 6}
    _, rectangle = calibration_of(built(name, **region), name)
    views, ellipse = calibration_of(built(name, elliptical_acs=True, **region), name)
    n_y, n_z = SMALL[name]["n_y"], SMALL[name]["n_z"]
    centre = (n_y // 2, n_z // 2)

    assert len(rectangle) == 8 * 6
    assert ellipse == {view for view in rectangle if inside(view, centre, (8, 6))}
    assert (n_y // 2 - 4, n_z // 2 - 3) not in ellipse
    assert centre in ellipse
    assert views[: len(ellipse)] == sorted(ellipse)


@pytest.mark.parametrize("name", SMALL)
def test_a_fully_sampled_scan_has_no_calibration_region_whatever_its_shape(name):
    seq = built(name, elliptical_acs=True, n_acs_y=4, n_acs_z=2)

    assert calibration_of(seq, name)[1] == set()


@pytest.mark.parametrize("name", SMALL)
def test_the_wave_free_reference_leads_and_is_marked_ref(name):
    prescription = {
        "wave": "both",
        "wave_cycles": 2,
        "wave_amplitude": 8e-3,
        "ry": 2,
        "n_acs_y": 4,
        "n_acs_z": 2,
    }
    seq = built(name, **prescription)
    lin, par, ref, ima, seg = adc_labels(seq, "LIN", "PAR", "REF", "IMA", "SEG")
    calibrating, imaging = sampled(name, **prescription)
    reference = sorted(calibrating)
    views = [
        v for v in [*reference, *calibrating, *imaging] for _ in range(echoes(name))
    ]
    n_reference = len(reference) * echoes(name)

    assert seq.check_timing()[0]
    assert list(zip(lin, par, strict=True)) == views
    assert list(ref) == [1] * n_reference + [0] * (len(views) - n_reference)
    assert list(seg) == [0] * n_reference + [1] * (len(views) - n_reference)
    assert set(ima) == {0}


@pytest.mark.parametrize("name", SMALL)
def test_the_reference_is_played_with_the_corkscrew_scaled_away(name):
    prescription = {
        "wave": "phase",
        "wave_cycles": 2,
        "wave_amplitude": 8e-3,
        "ry": 2,
        "n_acs_y": 4,
        "n_acs_z": 2,
    }
    reads = acquisitions(built(name, **prescription))
    n_reference = len(sampled(name, **prescription)[0]) * echoes(name)

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
    prescription = {"ry": 2, "n_acs_y": 4, "n_acs_z": 2, **wave}
    seq = built(name, **prescription)
    calibrating, imaging = sampled(name, **prescription)
    (ref,) = adc_labels(seq, "REF")

    assert len(acquisitions(seq)) == (len(calibrating) + len(imaging)) * echoes(name)
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
    seq = built(name, n_dummy=2, ry=2, n_acs_y=4, n_acs_z=2, **prescription)

    assert seq.repetition()[1] == 1


@pytest.mark.parametrize("name", SMALL)
def test_a_wave_amplitude_beyond_the_system_limits_is_played_at_the_one_admitted(name):
    wave = {"wave": "both", "wave_cycles": 8, "ry": 2, "n_acs_y": 4, "n_acs_z": 2}
    below = wave_waveform(built(name, wave_amplitude=1e-3, **wave))
    capped = wave_waveform(built(name, wave_amplitude=0.2, **wave))
    # Until a limit binds, the wave is the amplitude requested times a fixed shape.
    admitted = 1e-3 * np.max(np.abs(capped)) / np.max(np.abs(below))
    again = wave_waveform(built(name, wave_amplitude=admitted, **wave))

    assert 0.0 < admitted < 0.2
    assert wave_waveform(built(name, wave_amplitude=0.5, **wave)) == pytest.approx(
        capped
    )
    assert again == pytest.approx(capped)


@pytest.mark.parametrize("name", SMALL)
def test_an_unknown_wave_mode_is_refused(name):
    with pytest.raises(ValueError, match="wave mode"):
        built(name, wave="corkscrew")

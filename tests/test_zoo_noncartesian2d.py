"""The 2D non-Cartesian example sequences: radial, spiral and PROPELLER, gradient and spin echo."""

import importlib
import math
from itertools import pairwise

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per example sequence.
SMALL = {
    "gre_radial2D_sequence": {"n": 32, "tr": None},
    "gre_spiral2D_sequence": {"n": 32, "n_shots": 4, "tr": None},
    "gre_propeller2D_sequence": {"n": 32, "blade_width": 8, "tr": None},
    "se_radial2D_sequence": {"n": 32, "tr": None},
    "se_spiral2D_sequence": {"n": 32, "n_shots": 4, "tr": None},
    "se_propeller2D_sequence": {"n": 32, "blade_width": 8, "te": None, "tr": None},
    "se_epi_propeller2D_sequence": {
        "n_x": 32,
        "blade_width": 8,
        "n_blades": 4,
        "te": None,
        "tr": None,
    },
}

RADIAL = ["gre_radial2D_sequence", "se_radial2D_sequence"]
SPIRAL = ["gre_spiral2D_sequence", "se_spiral2D_sequence"]
PROPELLER = ["gre_propeller2D_sequence", "se_propeller2D_sequence"]
#: One tilt per excitation, played in order.
TILTED = RADIAL + SPIRAL + PROPELLER
SPIN_ECHO = ["se_radial2D_sequence", "se_spiral2D_sequence", "se_propeller2D_sequence"]

#: Every definition each entry writes.
DEFINITIONS = {
    **dict.fromkeys(
        RADIAL, frozenset({"NumSpokes", "kSpaceCenterSample", "SlicePositions"})
    ),
    **dict.fromkeys(
        SPIRAL, frozenset({"NumArms", "kSpaceCenterSample", "SlicePositions"})
    ),
    **dict.fromkeys(
        PROPELLER, frozenset({"BladeWidth", "NumBlades", "kSpaceCenterLine"})
    ),
    "se_epi_propeller2D_sequence": {
        "BladeWidth",
        "NumBlades",
        "NumGainCalibrationReadouts",
    },
}
COMMON = {"FOV", "Matrix", "Name", "TE", "TR", "Trajectory"}


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def app_class(name):
    mod = module(name)
    return next(
        value
        for key, value in vars(mod).items()
        if key.endswith("App") and getattr(value, "__module__", None) == mod.__name__
    )


def app(name, **kwargs):
    """The entry's application, built from its small prescription."""
    return app_class(name)(pp.Opts(), **{**SMALL[name], **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def blocks(seq):
    return [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]


def rotation_angle(block):
    """The in-plane angle a block's rotation extension turns by, in radians."""
    q = np.asarray(block.rotation.quaternion)
    return 2.0 * np.arctan2(q[3], q[0])


def same_angles(played, intended):
    """Equal modulo a full turn."""
    return np.allclose(
        np.exp(1j * np.asarray(played)), np.exp(1j * np.asarray(intended))
    )


def in_plane(block):
    return block.gx is not None or block.gy is not None


def tilts(built):
    """The tilt index of every excitation, dummies first, for one slice."""
    views = getattr(built, "views", None)
    played = [b for b, _ in views] if views else list(range(len(built.angles)))
    return [0] * built.n_dummy + played


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_builds_a_sequence_that_passes_its_timing_check(name):
    seq = module(name).main(**SMALL[name])

    is_ok, errors = seq.check_timing()
    assert is_ok, errors


@pytest.mark.parametrize("name", SMALL)
def test_every_definition_is_written(name):
    seq = module(name).main(**SMALL[name])

    assert COMMON | DEFINITIONS[name] <= set(seq.definitions)


# -- the tilts ---------------------------------------------------------------


@pytest.mark.parametrize("ry", [1, 2, 3])
@pytest.mark.parametrize(
    ("name", "nyquist", "span"),
    [
        *((name, math.ceil(np.pi / 2 * 32), np.pi) for name in RADIAL),
        *((name, 4, 2 * np.pi) for name in SPIRAL),
        *((name, math.ceil(np.pi * 32 / 16), np.pi) for name in PROPELLER),
    ],
)
def test_ry_plays_every_ryth_tilt_of_the_nyquist_set_in_order(name, nyquist, span, ry):
    built = app(name, ry=ry)

    assert built.angles == pytest.approx(span * np.arange(0, nyquist, ry) / nyquist)


@pytest.mark.parametrize("name", TILTED)
def test_every_in_plane_block_of_a_shot_is_turned_by_its_tilt(name):
    built = app(name, n_dummy=2)
    shots = iter(tilts(built))

    played, intended = [], []
    for block in blocks(built.design()):
        if block.rf is not None and block.rf.use == "excitation":
            angle = built.angles[next(shots)]
        elif in_plane(block):
            played.append(rotation_angle(block))
            intended.append(angle)

    assert next(shots, None) is None
    assert same_angles(played, intended)


@pytest.mark.parametrize("name", [*SPIN_ECHO, "se_epi_propeller2D_sequence"])
def test_the_refocusing_pulse_is_not_turned(name):
    seq = module(name).main(**SMALL[name])

    assert all(b.rotation is None for b in blocks(seq) if b.rf is not None)


# -- labels ------------------------------------------------------------------


@pytest.mark.parametrize("name", RADIAL + SPIRAL)
def test_each_acquisition_carries_its_shot_and_slice_in_play_order(name):
    built = app(name, n_dummy=2, n_slices=3)
    lin, slc = adc_labels(built.design(), "LIN", "SLC")

    expected = [
        (i, s)
        for packet in built.packets
        for i in range(len(built.angles))
        for s in packet
    ]
    assert list(zip(lin, slc, strict=True)) == expected


@pytest.mark.parametrize("name", PROPELLER)
def test_each_blade_line_carries_its_line_blade_and_slice_in_play_order(name):
    built = app(name, n_dummy=2, n_slices=3)
    lin, seg, slc = adc_labels(built.design(), "LIN", "SEG", "SLC")

    expected = [
        (line, blade, s)
        for packet in built.packets
        for blade, line in built.views
        for s in packet
    ]
    assert list(zip(lin, seg, slc, strict=True)) == expected


def test_each_line_of_an_epi_blade_carries_its_line_blade_and_slice():
    built = app("se_epi_propeller2D_sequence", n_slices=2, n_dummy=1)
    lin, slc, seg = adc_labels(built.design(), "LIN", "SLC", "SEG")

    width = built.blade.etl
    expected = [
        (line, s, b)
        for group in built.passes
        for b in range(built.blade.n_blades)
        for s in group
        for line in range(width)
    ]
    assert list(zip(lin, slc, seg, strict=True)) == expected


@pytest.mark.parametrize("scheme", ["uniform", "golden"])
def test_every_encoding_block_of_an_epi_blade_is_turned_by_its_angle(scheme):
    built = app("se_epi_propeller2D_sequence", angle_scheme=scheme, n_dummy=1)
    angles = built.blade.blade_angles

    shots = iter([0] * built.n_dummy + list(range(len(angles))))
    played, intended = [], []
    for block in blocks(built.design()):
        if block.rf is not None and block.rf.use == "excitation":
            angle = angles[next(shots)]
        elif in_plane(block):
            played.append(rotation_angle(block))
            intended.append(angle)

    assert next(shots, None) is None
    assert same_angles(played, intended)


# -- timing ------------------------------------------------------------------


def played(seq):
    """RF ``(centre time, use)`` and echo-sample times, in play order."""
    center = int(np.atleast_1d(seq.definitions["kSpaceCenterSample"])[0])
    t, pulses, echoes = 0.0, [], []
    for block in blocks(seq):
        if block.rf is not None:
            pulses.append((t + block.rf.delay + block.rf.center, block.rf.use))
        if block.adc is not None:
            echoes.append(t + block.adc.delay + center * block.adc.dwell)
        t += block.block_duration
    return pulses, echoes


@pytest.mark.parametrize("offset", [4e-3, 4.013e-3], ids=["on raster", "off raster"])
@pytest.mark.parametrize("name", SPIN_ECHO)
def test_a_spin_echo_samples_its_centre_where_the_180_refocuses(name, offset):
    """The 180 sits midway even for a TE off the raster, which is rounded up."""
    requested = app(name).echo_time + offset
    built = app(name, te=requested)
    seq = built.design()
    pulses, echoes = played(seq)
    (excitation, _), (refocusing, use) = pulses[:2]
    written = np.atleast_1d(seq.definitions["TE"])[0]
    raster = built.system.block_duration_raster

    assert use == "refocusing"
    assert echoes[0] - excitation == pytest.approx(written, abs=1e-9)
    assert refocusing - excitation == pytest.approx(written / 2, abs=1e-9)
    assert requested - 1e-9 <= written <= requested + 2 * raster


@pytest.mark.parametrize(
    "te", [None, 60e-3, 60.0047e-3], ids=["shortest", "on raster", "off raster"]
)
def test_an_epi_blade_reads_its_central_line_at_the_resolved_echo_time(te):
    built = app("se_epi_propeller2D_sequence", te=te)
    seq = built.design()
    k, _, t_excitation, _, t_adc = seq.calculate_kspace()
    n, centre = int(built.blade.adc.num_samples), built.blade.blade_width // 2
    kx = np.asarray(k)[0, centre * n : (centre + 1) * n]
    t = np.asarray(t_adc)[centre * n : (centre + 1) * n]
    # The first blade reads along x, and crosses k = 0 between two samples.
    i = int(np.flatnonzero(np.diff(np.sign(kx)))[0])
    crossing = t[i] - kx[i] * (t[i + 1] - t[i]) / (kx[i + 1] - kx[i])

    assert crossing - t_excitation[0] == pytest.approx(built.resolved["te"], abs=1e-9)
    assert np.atleast_1d(seq.definitions["TE"])[0] == pytest.approx(
        built.resolved["te"]
    )


def test_a_blade_count_left_to_the_design_resolves_to_the_nyquist_set_played():
    built = app("se_epi_propeller2D_sequence", n_blades=None)
    (seg,) = adc_labels(built.design(), "SEG")

    assert built.resolved["n_blades"] == math.ceil(np.pi * 32 / (2 * 8))
    assert sorted(set(seg)) == list(range(built.resolved["n_blades"]))


def test_a_gain_calibration_left_to_the_design_resolves_to_one_readout_per_slice():
    built = app("se_epi_propeller2D_sequence", n_slices=3)
    written = built.design().definitions["NumGainCalibrationReadouts"]

    assert built.resolved["n_gain_calibration_readouts"] == 3
    assert np.atleast_1d(written)[0] == 3


@pytest.mark.parametrize(
    ("name", "prescription"),
    [
        *((name, {"te": 1e-6}) for name in RADIAL[:1] + SPIRAL[:1] + PROPELLER[:1]),
        *((name, {"tr": 1e-4}) for name in RADIAL[:1] + SPIRAL[:1] + PROPELLER[:1]),
        *((name, {"te": 1e-3}) for name in SPIN_ECHO),
        *((name, {"tr": 1e-3}) for name in SPIN_ECHO),
        ("se_epi_propeller2D_sequence", {"te": 1e-3}),
        ("se_epi_propeller2D_sequence", {"tr": 1e-3}),
    ],
)
def test_an_infeasible_prescription_is_refused(name, prescription):
    with pytest.raises(ValueError, match="shorter than"):
        module(name).main(**{**SMALL[name], **prescription})


@pytest.mark.parametrize("name", TILTED)
def test_every_slice_is_excited_at_the_repetition_time_asked_for(name):
    tr = 3 * app(name).repetition_time
    built = app(name, n_dummy=0, n_slices=5, tr=tr)
    excited = np.asarray(built.design().rf_times()[0])
    shots = len(tilts(built))

    assert len(built.packets) > 1
    at = 0
    for packet in built.packets:
        spacing = np.diff(excited[at : at + len(packet) * shots][:: len(packet)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(packet) * shots


# -- every shot ----------------------------------------------------------------


def area(event):
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


@pytest.mark.parametrize("name", [*TILTED, "se_epi_propeller2D_sequence"])
def test_every_shot_closes_its_in_plane_gradient_moment(name):
    """A residual moment would turn with the shot and differ from one to the next."""
    built = app(name, n_dummy=1)
    seq = built.design()
    delta_k = 1.0 / built.fov

    # Nothing in-plane plays before a refocusing pulse, so a shot's moment is
    # the plain integral of the played physical waveforms between excitations.
    waveforms = seq.waveforms()
    excitations = np.asarray(seq.rf_times()[0])
    edges = [*excitations, seq.duration()[0]]

    def moment(axis, start, stop):
        t, g = (np.asarray(v, dtype=float) for v in waveforms[axis][:2])
        grid = np.unique(np.concatenate([t[(t > start) & (t < stop)], [start, stop]]))
        return np.trapezoid(np.interp(grid, t, g, left=0.0, right=0.0), grid)

    moments = [[moment(axis, a, b) for axis in (0, 1)] for a, b in pairwise(edges)]

    if "epi" in name:
        assert len(moments) == built.n_dummy + built.blade.n_blades
    else:
        assert len(moments) == len(tilts(built))
    assert np.abs(moments).max() < 1e-3 * delta_k


def layout(block):
    return (
        round(block.block_duration, 9),
        *(getattr(block, c) is not None for c in ("rf", "gx", "gy", "gz", "adc")),
        *(
            round(area(getattr(block, c)), 6)
            for c in ("gx", "gy", "gz")
            if getattr(block, c) is not None
        ),
    )


@pytest.mark.parametrize(
    ("name", "prescription"),
    [
        ("gre_radial2D_sequence", {"te": None}),
        ("gre_radial2D_sequence", {"te": 5e-3, "slice_thickness": 20e-3}),
        ("gre_spiral2D_sequence", {"te": None}),
        ("gre_spiral2D_sequence", {"te": 6e-3}),
        ("gre_spiral2D_sequence", {"density": "dual", "periphery_undersampling": 3}),
        ("se_radial2D_sequence", {"te": None}),
        ("se_spiral2D_sequence", {"te": None}),
    ],
)
def test_a_shot_plays_the_block_layout_its_readout_solved(name, prescription):
    """Same blocks, durations and gradient areas as the readout module's."""
    built = app(name, n_dummy=0, **prescription)
    seq = built.design()
    solved = [
        layout(built.ro.seq.get_block(i)) for i in range(2, len(built.ro.blocks) + 1)
    ]
    first = 2 if name.startswith("gre") else 4 + (built.wait_half_te is not None)

    played_layout = [
        layout(seq.get_block(i)) for i in range(first, first + len(solved))
    ]
    assert played_layout == solved


# -- spiral density ------------------------------------------------------------


@pytest.mark.parametrize("name", SPIRAL)
@pytest.mark.parametrize("density", ["variable", "dual"])
def test_a_sparser_periphery_shortens_the_interleave(name, density):
    constant = app(name).ro.trajectory.read_duration
    sparse = app(name, density=density, periphery_undersampling=3).ro

    assert sparse.trajectory.read_duration < constant


@pytest.mark.parametrize("name", SPIRAL)
@pytest.mark.parametrize(
    "prescription", [{"density": "logarithmic"}, {"periphery_undersampling": 0.5}]
)
def test_an_unknown_density_or_a_denser_periphery_is_refused(name, prescription):
    with pytest.raises(ValueError):
        app(name, **prescription)


@pytest.mark.parametrize("name", PROPELLER)
@pytest.mark.parametrize("blade_width", [0, 33])
def test_a_blade_wider_than_the_matrix_is_refused(name, blade_width):
    with pytest.raises(ValueError, match="blade_width"):
        app(name, blade_width=blade_width)


# -- the command line ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        ("gre_radial2D_sequence", "--readout-oversampling", "Readout oversampling"),
        ("gre_spiral2D_sequence", "--n-shots", "Interleaves that sample the centre"),
        ("se_propeller2D_sequence", "--blade-width", "Phase-encode lines per blade"),
        (
            "se_epi_propeller2D_sequence",
            "--blade-width",
            "Phase-encode lines per blade",
        ),
    ],
)
def test_a_flag_is_named_and_described_by_the_function_it_runs(
    capsys, name, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module(name).main, ["--help"])

    printed = capsys.readouterr().out

    assert flag in printed
    assert help_text in printed

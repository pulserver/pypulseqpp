"""3D non-Cartesian zoo entries: stacks of stars, spirals and blades, and ZTE."""

import importlib
from itertools import pairwise

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per entry.
SMALL = {
    "gre_stack_of_stars3D_sequence": {"n": 32, "n_z": 4},
    "gre_stack_of_spirals3D_sequence": {"n": 32, "n_z": 4, "n_shots": 4},
    "gre_stack_of_blades3D_sequence": {"n": 32, "n_z": 4, "blade_width": 8},
    "se_stack_of_stars3D_sequence": {"n": 32, "n_z": 4, "tr": None},
    "se_stack_of_spirals3D_sequence": {"n": 32, "n_z": 4, "n_shots": 4, "tr": None},
    "se_stack_of_blades3D_sequence": {
        "n": 32,
        "n_z": 4,
        "blade_width": 8,
        "tr": None,
    },
    "zte3D_sequence": {"n_x": 32, "n_views": 24, "n_shots": 2, "n_dummy": 0},
}
GRE = [
    "gre_stack_of_stars3D_sequence",
    "gre_stack_of_spirals3D_sequence",
    "gre_stack_of_blades3D_sequence",
]
SPIN_ECHO = [
    "se_stack_of_stars3D_sequence",
    "se_stack_of_spirals3D_sequence",
    "se_stack_of_blades3D_sequence",
]
STACKS = GRE + SPIN_ECHO
#: Stacks whose readout plays one arm per excitation, rather than a blade line.
ARMED = [name for name in STACKS if "blades" not in name]
BLADES = [name for name in STACKS if "blades" in name]


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def app_class(name):
    mod = module(name)
    return next(
        value
        for key, value in vars(mod).items()
        if key.endswith("App") and getattr(value, "__module__", None) == mod.__name__
    )


def app(name, dummies=None, **kwargs):
    """The entry's application, with ``dummies`` non-acquiring repetitions if given."""
    cls = app_class(name)
    if dummies is not None:
        cls = type(cls.__name__, (cls,), {"N_DUMMY": dummies})
    return cls(pp.Opts(), **{**SMALL[name], **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def blocks(seq):
    return [seq.get_block(index) for index in range(1, len(seq.block_events) + 1)]


def acquisitions(seq):
    """Every block that acquires, in play order."""
    return [block for block in blocks(seq) if block.adc is not None]


def excitations(seq):
    return [b for b in blocks(seq) if b.rf is not None and b.rf.use == "excitation"]


def z_angle(quaternion):
    """Angle of a scalar-first quaternion that turns about z, in radians."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    assert x == pytest.approx(0.0, abs=1e-12)
    assert y == pytest.approx(0.0, abs=1e-12)
    return 2.0 * np.arctan2(z, w)


def matrix_of(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def area(event):
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


def shot_views(built):
    """``(tilt, partition)`` of every excitation, dummies first."""
    n_z = built.matrix[2]
    if "blades" in built.NAME:
        played = [(blade, z) for blade, _, z in built.views]
    else:
        played = list(built.views)
    return [(0, n_z // 2)] * built.N_DUMMY + played


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_builds_a_sequence_that_passes_its_timing_check(name):
    seq = module(name).main(**SMALL[name])

    is_ok, errors = seq.check_timing()
    assert is_ok, errors
    assert seq.definitions["Name"] == app_class(name).NAME


@pytest.mark.parametrize("excitation", ["slab", "nonselective", "spsp"])
@pytest.mark.parametrize("name", STACKS)
def test_every_excitation_builds_a_stack_that_passes_its_timing_check(name, excitation):
    seq = module(name).main(**SMALL[name], excitation=excitation)

    assert seq.check_timing()[0]
    assert seq.definitions["Excitation"] == excitation


@pytest.mark.parametrize("name", STACKS)
def test_every_stack_definition_is_written(name):
    seq = module(name).main(**SMALL[name])

    assert {
        "FOV",
        "Matrix",
        "TE",
        "TR",
        "Trajectory",
        "PartitionAngleShift",
        "kSpaceCenterPartition",
        "kSpaceCenterSample",
    } <= set(seq.definitions)


# -- what is sampled -----------------------------------------------------------


@pytest.mark.parametrize("ry", [1, 2, 3])
@pytest.mark.parametrize(
    ("name", "nyquist", "span"),
    [
        *((n, int(np.ceil(np.pi / 2 * 32)), np.pi) for n in STACKS if "stars" in n),
        *((n, 4, 2 * np.pi) for n in STACKS if "spirals" in n),
        *((n, int(np.ceil(np.pi * 32 / 16)), np.pi) for n in BLADES),
    ],
)
def test_ry_plays_every_ryth_tilt_of_the_nyquist_set_in_order(name, nyquist, span, ry):
    built = app(name, ry=ry)

    assert built.angles == pytest.approx(span * np.arange(0, nyquist, ry) / nyquist)


@pytest.mark.parametrize("name", STACKS)
def test_rz_and_partial_fourier_keep_the_centre_partition_and_drop_the_early_ones(
    name,
):
    built = app(name, n_z=8, rz=2, partial_fourier_z=0.75, n_acs_z=0)

    assert built.partitions == [2, 4, 6]


@pytest.mark.parametrize("name", STACKS)
def test_the_calibration_partitions_lead_at_every_tilt_and_are_marked_ima(name):
    stack = app(name, dummies=0, n_z=16, rz=4, n_acs_z=4)
    par, ima = adc_labels(stack.design(), "PAR", "IMA")
    leading = [view for view in stack.views if view[-1] in stack.calibration]

    assert stack.calibration == {6, 7, 8, 9}
    assert stack.partitions == [0, 4, 6, 7, 8, 9, 12]
    assert stack.views[: len(leading)] == leading
    assert list(par) == [view[-1] for view in stack.views]
    assert list(ima) == [int(p in stack.calibration) for p in par]


@pytest.mark.parametrize("name", STACKS)
def test_a_fully_sampled_stack_has_no_calibration_partitions(name):
    stack = app(name, rz=1, n_acs_z=4)

    assert stack.calibration == set()
    assert stack.partitions == list(range(stack.matrix[2]))


@pytest.mark.parametrize("name", ARMED)
def test_every_partition_of_a_stack_arm_is_acquired_before_the_next_arm(name):
    stack = app(name, dummies=3, rz=2)
    seq = stack.design()
    lin, par, once = adc_labels(seq, "LIN", "PAR", "ONCE")

    expected = list(stack.views)
    assert list(zip(lin, par, strict=True)) == expected
    assert set(once) == {0}
    assert len(excitations(seq)) == 3 + len(expected)


@pytest.mark.parametrize("name", BLADES)
def test_every_partition_of_a_blade_line_is_acquired_before_the_next_line(name):
    stack = app(name, dummies=3, rz=2)
    seq = stack.design()
    lin, seg, par = adc_labels(seq, "LIN", "SEG", "PAR")

    expected = [(line, blade, z) for blade, line, z in stack.views]
    assert list(zip(lin, seg, par, strict=True)) == expected
    assert len(excitations(seq)) == 3 + len(expected)


@pytest.mark.parametrize("shift", ["none", "golden", "tiny_golden"])
@pytest.mark.parametrize("name", STACKS)
def test_every_in_plane_block_is_turned_to_its_tilt_and_partition_angle(name, shift):
    stack = app(name, dummies=1, partition_angle_shift=shift)
    fraction = module(name).PARTITION_SHIFTS[shift]
    views = iter(shot_views(stack))

    played, intended = [], []
    for block in blocks(stack.design()):
        if block.rf is not None and block.rf.use == "excitation":
            tilt, partition = next(views)
            angle = stack.angles[tilt] + partition * fraction * stack.span
        elif block.gx is not None or block.gy is not None:
            played.append(z_angle(block.rotation.quaternion))
            intended.append(angle)

    assert next(views, None) is None
    turns = 2 * np.pi / stack.span
    assert np.exp(1j * turns * np.asarray(played)) == pytest.approx(
        np.exp(1j * turns * np.asarray(intended)), abs=1e-9
    )


@pytest.mark.parametrize("name", GRE)
def test_every_partition_is_encoded_at_the_step_its_label_names(name):
    """The z area played between the excitation and the samples is the partition's."""
    stack = app(name, n_z=8)
    seq = stack.design()
    n_z = stack.matrix[2]
    delta_kz = 1.0 / stack.fov_z

    encoded, area_since = [], None
    for block in blocks(seq):
        if block.rf is not None:
            area_since = 0.0
            continue
        if block.gz is not None and area_since is not None:
            area_since += area(block.gz)
        if block.adc is not None:
            encoded.append(area_since)
            area_since = None
    (par,) = adc_labels(seq, "PAR")

    assert encoded == pytest.approx((par - n_z // 2) * delta_kz, abs=1e-6 * delta_kz)


@pytest.mark.parametrize("name", STACKS)
def test_every_shot_closes_its_in_plane_gradient_moment(name):
    """A residual moment would turn with the shot and differ from one to the next."""
    stack = app(name, dummies=1)
    seq = stack.design()
    delta_k = 1.0 / stack.fov

    waveforms = seq.waveforms()
    starts, t = [], 0.0
    for block in blocks(seq):
        if block.rf is not None and block.rf.use == "excitation":
            starts.append(t)
        t += block.block_duration
    edges = [*starts, seq.duration()[0]]

    def moment(axis, start, stop):
        t, g = (np.asarray(v, dtype=float) for v in waveforms[axis][:2])
        grid = np.unique(np.concatenate([t[(t > start) & (t < stop)], [start, stop]]))
        return np.trapezoid(np.interp(grid, t, g, left=0.0, right=0.0), grid)

    moments = [[moment(axis, a, b) for axis in (0, 1)] for a, b in pairwise(edges)]

    assert np.abs(moments).max() < 1e-3 * delta_k


# -- timing ----------------------------------------------------------------------


def played(seq):
    """RF centre times and the first echo-sample time, in play order."""
    center = int(np.atleast_1d(seq.definitions["kSpaceCenterSample"])[0])
    t, pulses = 0.0, []
    for block in blocks(seq):
        if block.rf is not None:
            pulses.append((t + block.rf.delay + block.rf.center, block.rf.use))
        if block.adc is not None:
            return pulses, t + block.adc.delay + center * block.adc.dwell
        t += block.block_duration
    raise AssertionError("no acquisition")


@pytest.mark.parametrize("excitation", ["slab", "nonselective", "spsp"])
@pytest.mark.parametrize("name", SPIN_ECHO)
def test_a_spin_echo_samples_its_centre_where_the_180_refocuses(name, excitation):
    """The 180 sits midway even for a TE off the raster, which is rounded up."""
    requested = app(name, excitation=excitation).echo_time + 4.013e-3
    built = app(name, excitation=excitation, te=requested)
    seq = built.design()
    pulses, echo = played(seq)
    (excitation_time, _), (refocusing_time, use) = pulses[:2]
    written = np.atleast_1d(seq.definitions["TE"])[0]
    raster = built.system.block_duration_raster

    assert use == "refocusing"
    assert echo - excitation_time == pytest.approx(written, abs=1e-9)
    assert refocusing_time - excitation_time == pytest.approx(written / 2, abs=1e-9)
    assert requested - 1e-9 <= written <= requested + 2 * raster


@pytest.mark.parametrize(
    ("name", "prescription"),
    [
        *((name, {"te": 1e-6}) for name in GRE),
        *((name, {"tr": 1e-4}) for name in GRE),
        *((name, {"te": 1e-3}) for name in SPIN_ECHO),
        *((name, {"tr": 1e-3}) for name in SPIN_ECHO),
    ],
)
def test_an_echo_or_repetition_shorter_than_the_shot_is_refused(name, prescription):
    with pytest.raises(ValueError, match="shorter than"):
        module(name).main(**{**SMALL[name], **prescription})


@pytest.mark.parametrize(
    "prescription",
    [
        {"excitation": "adiabatic"},
        {"partition_angle_shift": "random"},
        {"rz": 0},
        {"partial_fourier_z": 0.5},
    ],
    ids=str,
)
@pytest.mark.parametrize("name", STACKS)
def test_an_unknown_choice_or_an_out_of_range_factor_is_refused(name, prescription):
    with pytest.raises(ValueError):
        app(name, **prescription)


# -- ZTE ---------------------------------------------------------------------------


def test_every_zte_view_of_every_shot_is_acquired_once_in_order():
    zte = app("zte3D_sequence", n_shots=3, n_dummy=2)
    lin, seg, once = adc_labels(zte.design(), "LIN", "SEG", "ONCE")
    n_views = len(zte.zte.directions)

    assert list(zip(lin, seg, strict=True)) == [
        (view, shot) for shot in range(3) for view in range(n_views)
    ]
    assert set(once) == {0}


def test_every_zte_acquisition_is_turned_by_its_shot_rotation():
    zte = app("zte3D_sequence", n_shots=3, n_dummy=1)
    seq = zte.design()
    (seg,) = adc_labels(seq, "SEG")

    for block, shot in zip(acquisitions(seq), seg, strict=True):
        assert matrix_of(block.rotation.quaternion) == pytest.approx(
            zte.zte.shot_rotations[shot], abs=1e-9
        )
    # A dummy shell is the first shot without the ADC.
    first = seq.get_block(1).rotation.quaternion
    assert matrix_of(first) == pytest.approx(zte.zte.shot_rotations[0], abs=1e-9)


def test_a_zte_pulse_too_long_for_its_dead_time_is_refused():
    with pytest.raises(ValueError):
        module("zte3D_sequence").main(**SMALL["zte3D_sequence"], pulse_duration=1e-3)


# -- the command line ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        (
            "gre_stack_of_stars3D_sequence",
            "--partition-angle-shift",
            "How far each partition turns",
        ),
        ("se_stack_of_spirals3D_sequence", "--n-shots", "Interleaves that sample"),
        ("gre_stack_of_blades3D_sequence", "--blade-width", "Phase-encode lines"),
        ("zte3D_sequence", "--n-views", "Views per shell."),
    ],
)
def test_a_flag_is_named_and_described_by_the_function_it_runs(
    capsys, name, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module(name).main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())

    assert flag in printed
    assert help_text in printed

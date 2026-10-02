"""3D non-Cartesian example sequences: stacks of stars, spirals and blades, and ZTE."""

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
    "zte3D_sequence": {"n": 32, "n_shots": 2, "n_dummy": 0},
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


def built(name, **kwargs):
    """The sequence ``name`` designs from its small prescription."""
    return module(name).main(**{**SMALL[name], **kwargs})


def definition(seq, key):
    """The first value of a definition."""
    return np.atleast_1d(seq.definitions[key])[0]


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


def about_z(angle):
    """The matrix of a turn by ``angle`` (radians) about z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def phasors(angles, span):
    """``exp(2j * pi * angle / span)``: angles a whole span apart coincide."""
    return np.exp(2j * np.pi * np.asarray(angles) / span)


def area(event):
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


def span_of(name):
    """The turn the tilts of ``name`` cover: all of it for an interleaf, half otherwise."""
    return 2 * np.pi if "spirals" in name else np.pi


def tilt_label(name):
    """The label that indexes the tilt of an acquisition: the blade, else the arm."""
    return "SEG" if "blades" in name else "LIN"


def n_tilts(seq):
    """The tilts a stack plays, from the definition that counts them."""
    key = next(
        key for key in ("NumSpokes", "NumArms", "NumBlades") if key in seq.definitions
    )
    return int(definition(seq, key))


def views_per_partition(seq):
    """The acquisitions at each partition: an arm, or a line of a blade."""
    if "BladeWidth" in seq.definitions:
        return n_tilts(seq) * int(definition(seq, "BladeWidth"))
    return n_tilts(seq)


def n_z_of(seq):
    """The number of partitions, from the ``Matrix`` definition."""
    return int(np.atleast_1d(seq.definitions["Matrix"])[2])


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_builds_a_sequence_that_passes_its_timing_check(name):
    seq = built(name)

    is_ok, errors = seq.check_timing()
    assert is_ok, errors
    assert seq.definitions["Name"] == module(name).NAME


@pytest.mark.parametrize("excitation", ["slab", "nonselective", "spsp"])
@pytest.mark.parametrize("name", STACKS)
def test_every_excitation_builds_a_stack_that_passes_its_timing_check(name, excitation):
    seq = built(name, excitation=excitation)

    assert seq.check_timing()[0]
    assert seq.definitions["Excitation"] == excitation


@pytest.mark.parametrize("name", STACKS)
def test_every_stack_definition_is_written(name):
    seq = built(name)

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
    seq = built(name, ry=ry, n_dummy=0)
    angles = [z_angle(block.rotation.quaternion) for block in acquisitions(seq)]
    played = phasors(angles, span)
    # A tilt is played at every partition, and a blade at every line, in turn.
    distinct = played[np.r_[True, np.abs(np.diff(played)) > 1e-6]]
    intended = span * np.arange(0, nyquist, ry) / nyquist

    assert n_tilts(seq) == len(intended)
    assert distinct == pytest.approx(phasors(intended, span), abs=1e-9)


@pytest.mark.parametrize("name", STACKS)
def test_rz_and_partial_fourier_keep_the_centre_partition_and_drop_the_early_ones(
    name,
):
    seq = built(name, n_dummy=0, n_z=8, rz=2, partial_fourier_z=0.75, n_acs_z=0)
    (par,) = adc_labels(seq, "PAR")

    assert sorted(set(par)) == [2, 4, 6]


@pytest.mark.parametrize("name", STACKS)
def test_the_calibration_partitions_lead_at_every_tilt_and_are_marked_ima(name):
    seq = built(name, n_dummy=0, n_z=16, rz=4, n_acs_z=4)
    par, ima = adc_labels(seq, "PAR", "IMA")
    n_views = views_per_partition(seq)

    assert sorted(set(par)) == [0, 4, 6, 7, 8, 9, 12]
    assert len(par) == 7 * n_views
    # Each tilt, or blade line, takes the calibration partitions in turn, then
    # the periphery's.
    assert (par[: 4 * n_views].reshape(n_views, 4) == [6, 7, 8, 9]).all()
    assert (par[4 * n_views :].reshape(n_views, 3) == [0, 4, 12]).all()
    assert list(ima) == [1] * (4 * n_views) + [0] * (3 * n_views)


@pytest.mark.parametrize("name", STACKS)
def test_a_fully_sampled_stack_has_no_calibration_partitions(name):
    seq = built(name, n_dummy=0, rz=1, n_acs_z=4)
    par, ima = adc_labels(seq, "PAR", "IMA")

    assert not ima.any()
    assert sorted(set(par)) == list(range(n_z_of(seq)))


@pytest.mark.parametrize("name", ARMED)
def test_every_partition_of_a_stack_arm_is_acquired_before_the_next_arm(name):
    seq = built(name, n_dummy=3, n_z=8, rz=2, n_acs_z=2)
    lin, par, once = adc_labels(seq, "LIN", "PAR", "ONCE")
    calibrating, imaging = pp.make_cartesian_axis_sampling(8, 2, 2, partial_fourier=1.0)

    assert calibrating and imaging
    expected = [
        (arm, z)
        for partitions in (calibrating, imaging)
        for arm in range(n_tilts(seq))
        for z in partitions
    ]
    assert list(zip(lin, par, strict=True)) == expected
    assert set(once) == {0}
    assert len(excitations(seq)) == 3 + len(expected)


@pytest.mark.parametrize("name", BLADES)
def test_every_partition_of_a_blade_line_is_acquired_before_the_next_line(name):
    seq = built(name, n_dummy=3, n_z=8, rz=2, n_acs_z=2)
    lin, seg, par = adc_labels(seq, "LIN", "SEG", "PAR")
    calibrating, imaging = pp.make_cartesian_axis_sampling(8, 2, 2, partial_fourier=1.0)

    assert calibrating and imaging
    expected = [
        (line, blade, z)
        for partitions in (calibrating, imaging)
        for blade in range(n_tilts(seq))
        for line in range(int(definition(seq, "BladeWidth")))
        for z in partitions
    ]
    assert list(zip(lin, seg, par, strict=True)) == expected
    assert len(excitations(seq)) == 3 + len(expected)


@pytest.mark.parametrize("shift", ["none", "golden", "tiny_golden"])
@pytest.mark.parametrize("name", STACKS)
def test_every_in_plane_block_is_turned_to_its_tilt_and_partition_angle(name, shift):
    seq = built(name, n_dummy=1, partition_angle_shift=shift)
    span, fraction = span_of(name), module(name).PARTITION_SHIFTS[shift]
    tilt, partition = adc_labels(seq, tilt_label(name), "PAR")
    # The dummy plays first, at the first tilt and the centre partition.
    views = iter([(0, n_z_of(seq) // 2), *zip(tilt, partition, strict=True)])

    played, intended = [], []
    for block in blocks(seq):
        if block.rf is not None and block.rf.use == "excitation":
            index, z = next(views)
            angle = span * (index / n_tilts(seq) + z * fraction)
        elif block.gx is not None or block.gy is not None:
            played.append(z_angle(block.rotation.quaternion))
            intended.append(angle)

    assert next(views, None) is None
    assert phasors(played, span) == pytest.approx(phasors(intended, span), abs=1e-9)


@pytest.mark.parametrize("name", GRE)
def test_every_partition_is_encoded_at_the_step_its_label_names(name):
    """The z area played between the excitation and the samples is the partition's."""
    seq = built(name, n_z=8)
    n_z = n_z_of(seq)
    delta_kz = 1.0 / np.atleast_1d(seq.definitions["FOV"])[2]
    (par,) = adc_labels(seq, "PAR")

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

    assert encoded == pytest.approx((par - n_z // 2) * delta_kz, abs=1e-6 * delta_kz)


@pytest.mark.parametrize("name", STACKS)
def test_every_shot_closes_its_in_plane_gradient_moment(name):
    """A residual moment would turn with the shot and differ from one to the next."""
    seq = built(name, n_dummy=1)
    delta_k = 1.0 / np.atleast_1d(seq.definitions["FOV"])[0]

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
    requested = definition(built(name, excitation=excitation), "TE") + 4.013e-3
    seq = built(name, excitation=excitation, te=requested)
    pulses, echo = played(seq)
    (excitation_time, _), (refocusing_time, use) = pulses[:2]
    written = definition(seq, "TE")
    raster = pp.Opts().block_duration_raster

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
        built(name, **prescription)


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
        built(name, **prescription)


# -- ZTE ---------------------------------------------------------------------------


def test_every_zte_view_of_every_shot_is_acquired_once_in_order():
    seq = built("zte3D_sequence", n_shots=3, n_dummy=2)
    lin, seg, once = adc_labels(seq, "LIN", "SEG", "ONCE")
    n_views = int(definition(seq, "ViewsPerShot"))

    assert list(zip(lin, seg, strict=True)) == [
        (view, shot) for shot in range(3) for view in range(n_views)
    ]
    assert set(once) == {0}


def test_every_zte_acquisition_is_turned_by_its_shot_rotation():
    """Each shot is the first turned about z by a whole share of the turn."""
    seq = built("zte3D_sequence", n_shots=3, n_dummy=1)
    (seg,) = adc_labels(seq, "SEG")

    assert sorted(set(seg)) == [0, 1, 2]
    for block, shot in zip(acquisitions(seq), seg, strict=True):
        assert matrix_of(block.rotation.quaternion) == pytest.approx(
            about_z(2 * np.pi * shot / 3), abs=1e-9
        )
    # A dummy shell is the first shot without the ADC.
    first = seq.get_block(1).rotation.quaternion
    assert matrix_of(first) == pytest.approx(about_z(0.0), abs=1e-9)


def test_a_zte_pulse_too_long_for_its_dead_time_is_refused(monkeypatch):
    """The pulse duration is a module constant, so changing it reaches the refusal."""
    monkeypatch.setattr(module("zte3D_sequence"), "HARD_PULSE_DURATION", 1e-3)

    with pytest.raises(ValueError):
        built("zte3D_sequence")


def test_playing_one_zte_shell_in_every_r_acquires_that_fraction_of_the_shots():
    every = built("zte3D_sequence", n_shots=4, n_dummy=0)
    every_other = built("zte3D_sequence", n_shots=4, r=2, n_dummy=0)

    assert definition(every, "NumShots") == 4
    assert definition(every_other, "NumShots") == 2
    (seg,) = adc_labels(every, "SEG")
    assert sorted(set(seg)) == [0, 1, 2, 3]
    (seg,) = adc_labels(every_other, "SEG")
    assert sorted(set(seg)) == [0, 2]


def test_a_zte_shell_count_left_to_the_design_is_written_as_the_shells_played():
    """The shells balance the spacing within one against the spacing between them."""
    seq = built("zte3D_sequence", n=16, n_shots=None)
    (seg,) = adc_labels(seq, "SEG")
    n_shots = int(np.ceil(np.pi * (16 - 1)))

    assert sorted(set(seg)) == list(range(n_shots))
    assert definition(seq, "NumShots") == n_shots


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
        ("zte3D_sequence", "--scheme", "Shape of the shell"),
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

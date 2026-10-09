"""The balanced SSFP example sequences: 2D (cine gating) and 3D (phase-cycled chain)."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

bssfp2d = sequences.bssfp2D_sequence
bssfp3d = sequences.bssfp3D_sequence

SMALL_2D = {"n_x": 64, "n_y": 16, "readout_bandwidth_hz": 50e3, "n_dummy": 2}
SMALL_3D = {"n_x": 64, "n_y": 16, "n_z": 4}

#: A fast heart, so a gated train stays short.
CINE = {"views_per_segment": 4, "heart_rate_bpm": 300}


def built_2d(**kwargs):
    """The 2D sequence the small prescription designs, with ``kwargs`` changed."""
    return bssfp2d.main(**{**SMALL_2D, **kwargs})


def chain_3d(**kwargs):
    """The sequences the small 3D prescription designs, in play order, with ``kwargs`` changed."""
    return bssfp3d.main(**{**SMALL_3D, **kwargs})


def main_3d(**kwargs):
    """The last cycle's train of the small 3D prescription, with ``kwargs`` changed."""
    return chain_3d(**kwargs)[-1]


def recorded(seq, key):
    """The first value of the definition ``key`` of ``seq``."""
    return np.atleast_1d(seq.definitions[key])[0]


def labels(seq, *names, evolution="adc"):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution=evolution)
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def blocks(seq):
    return [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]


def excitations(seq):
    """``(time, phase, peak)`` of every RF pulse, in play order."""
    t, rows = 0.0, []
    for block in blocks(seq):
        if block.rf is not None:
            peak = np.abs(np.asarray(block.rf.signal)).max()
            rows.append(
                (t + block.rf.delay + block.rf.center, block.rf.phase_offset, peak)
            )
        t += block.block_duration
    return rows


def same_phase(a, b):
    return np.allclose(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))), 0.0)


def final_moments(seq):
    """Where the whole sequence leaves each gradient axis (1/m)."""
    return np.abs(np.asarray(seq.calculate_kspace()[1])[:, -1])


# -- 2D ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prescription",
    [
        {},
        {"gating": "retrospective", **CINE},
        {"gating": "prospective", "n_phases": 3, **CINE},
    ],
    ids=["ungated", "retrospective", "prospective"],
)
def test_each_slice_is_one_repetition_of_the_scan(prescription):
    seq = built_2d(n_slices=3, slice_spacing=1e-3, **prescription)
    size, start = seq.repetition()

    assert seq.check_timing()[0]
    assert start == 1
    assert 3 * size == len(seq.block_events)


def test_once_marks_each_slice_half_flip_and_closing_rewind():
    seq = built_2d(n_slices=2)
    (once,) = labels(seq, "ONCE", evolution="blocks")

    for chunk in np.split(once, 2):
        assert chunk[0] == 1
        assert chunk[-1] == 2
        assert set(chunk[1:-1]) == {0}


def test_the_phase_alternates_after_an_opposite_half_flip():
    seq = built_2d(n_slices=2, slice_spacing=1e-3)
    tr = np.atleast_1d(seq.definitions["TR"])[0]
    rows = excitations(seq)
    per_slice = len(rows) // 2
    offsets = [time for time, _, _ in rows]
    peaks = [peak for _, _, peak in rows]

    for start in (0, per_slice):
        train = rows[start : start + per_slice]
        centre = train[1][1] - np.pi
        assert same_phase(
            [phase - centre for _, phase, _ in train],
            [0.0] + [np.pi * ((k + 1) % 2) for k in range(per_slice - 1)],
        )
        assert peaks[start] == pytest.approx(0.5 * peaks[start + 1])
        assert offsets[start + 1] - offsets[start] == pytest.approx(0.5 * tr, abs=1e-9)


@pytest.mark.parametrize("ry", [1, 2])
def test_every_line_of_every_slice_is_read_once_when_ungated(ry):
    seq = built_2d(n_slices=2, ry=ry, n_acs_y=4)
    lin, slc, ima = labels(seq, "LIN", "SLC", "IMA")
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        16, ry, 4, partial_fourier=1.0
    )

    assert list(zip(lin, slc, strict=True)) == [
        (line, s) for s in range(2) for line in [*calibrating, *imaging]
    ]
    assert list(ima) == [int(line in calibrating) for line in lin]


def test_a_retrospective_heartbeat_cycles_its_segment():
    seq = built_2d(gating="retrospective", **CINE)
    lin, seg, phs = labels(seq, "LIN", "SEG", "PHS")
    cycles = max(1, round(0.2 / (4 * np.atleast_1d(seq.definitions["TR"])[0])))
    expected = [
        (line, s, c)
        for s, start in enumerate(range(0, 16, 4))
        for c in range(cycles)
        for line in range(start, start + 4)
    ]

    assert cycles > 1
    assert list(zip(lin, seg, phs, strict=True)) == expected
    assert not any(getattr(block, "trig", None) for block in blocks(seq))


def test_a_retrospective_cycle_reads_every_line_when_the_last_segment_is_short():
    seq = built_2d(gating="retrospective", **{**CINE, "views_per_segment": 6})
    lin, phs = labels(seq, "LIN", "PHS")

    for cycle in set(phs):
        assert sorted(lin[phs == cycle]) == list(range(16))


def test_a_prospective_heartbeat_waits_for_its_trigger_then_reads_every_phase():
    seq = built_2d(gating="prospective", n_phases=3, trigger_delay=5e-3, **CINE)
    lin, seg, phs = labels(seq, "LIN", "SEG", "PHS")
    triggers = [block.trig for block in blocks(seq) if getattr(block, "trig", None)]
    expected = [
        (line, s, f)
        for s, start in enumerate(range(0, 16, 4))
        for f in range(3)
        for line in range(start, start + 4)
    ]

    assert list(zip(lin, seg, phs, strict=True)) == expected
    assert len(triggers) == 4
    assert all(t[0].channel == "physio1" for t in triggers)
    assert all(t[0].duration == pytest.approx(5e-3) for t in triggers)
    assert seq.definitions["CardiacPhases"] == pytest.approx([3])


@pytest.mark.parametrize("n_dummy", [0, 3])
def test_the_first_trigger_of_a_slice_precedes_its_half_flip(n_dummy):
    seq = built_2d(
        gating="prospective", n_phases=2, n_slices=2, n_dummy=n_dummy, **CINE
    )
    listed = blocks(seq)
    (once,) = labels(seq, "ONCE", evolution="blocks")
    size = len(listed) // 2

    for start in (0, size):
        assert getattr(listed[start], "trig", None)
        assert once[start] == 1
        half, full = listed[start + 1].rf, listed[start + 4].rf
        peak = np.abs(np.asarray(half.signal)).max()
        assert peak == pytest.approx(0.5 * np.abs(np.asarray(full.signal)).max())


def test_a_single_shot_slice_has_one_trigger():
    seq = built_2d(gating="prospective", n_phases=1, views_per_segment=16, n_slices=2)
    triggers = [b for b in blocks(seq) if getattr(b, "trig", None)]

    assert seq.definitions["NumSegments"] == pytest.approx([1])
    assert len(triggers) == 2


def test_every_trigger_falls_between_two_balanced_repetitions():
    seq = built_2d(gating="prospective", n_phases=2, **CINE)
    listed = blocks(seq)
    for index, block in enumerate(listed):
        if getattr(block, "trig", None):
            assert block.gx is None and block.gy is None and block.gz is None
            assert listed[index + 1].rf is not None
    assert final_moments(seq)[:2] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("prescription", "match"),
    [
        ({"gating": "gated"}, "gating"),
        ({"gating": "prospective", "n_phases": 60, **CINE}, "heartbeat"),
        (
            {"gating": "retrospective", "heart_rate_bpm": 20, "views_per_segment": 2},
            "longer than",
        ),
        ({"tr": 1e-3}, "TR"),
    ],
    ids=["gating", "too many phases", "too long", "short TR"],
)
def test_a_2d_train_that_cannot_be_played_is_refused(prescription, match):
    with pytest.raises(ValueError, match=match):
        built_2d(**prescription)


# -- 3D ------------------------------------------------------------------------


@pytest.mark.parametrize("n_cycles", [1, 2, 3])
def test_the_chain_alternates_catalysts_and_cycles(n_cycles):
    chain = chain_3d(n_phase_cycles=n_cycles)

    assert [seq.definitions["Name"] for seq in chain] == [
        "bssfp_3d_catalyst",
        "bssfp_3d",
    ] * n_cycles
    assert [recorded(seq, "PhaseCycle") for seq in chain] == [
        k for k in range(n_cycles) for _ in range(2)
    ]
    for seq in chain:
        assert seq.check_timing()[0]
        assert seq.repetition()[1] == 1


def test_the_chain_is_written_as_files_linked_through_next_sequence(tmp_path):
    chain = chain_3d()
    paths = sequences.write(tmp_path / "scan.seq", chain)
    files = []
    for path in paths:
        seq = pp.Sequence(system=chain[0].system)
        seq.read(path)
        files.append(seq)

    assert [Path(p).name for p in paths] == ["scan.seq", "scan_bssfp_3d.seq"]
    assert [seq.definitions["Name"] for seq in files] == [
        "bssfp_3d_catalyst",
        "bssfp_3d",
    ]
    assert files[0].definitions["NextSequence"] == "scan_bssfp_3d.seq"
    assert "NextSequence" not in files[1].definitions
    for seq in files:
        assert seq.check_timing()[0]
        assert seq.repetition()[1] == 1


def test_a_tr_off_the_raster_is_written_as_the_tr_the_excitations_are_spaced_by():
    """The TR moves from the shortest in steps of two rasters, keeping TE = TR/2 on it."""
    shortest = recorded(main_3d(), "TR")
    seq = main_3d(tr=shortest + 63.3e-6)
    tr = recorded(seq, "TR")
    times = [time for time, _, _ in excitations(seq)]
    steps = (tr - shortest) / (2 * seq.system.block_duration_raster)

    assert np.diff(times) == pytest.approx(tr, abs=1e-9)
    assert steps == pytest.approx(round(steps))
    assert round(steps) > 0


@pytest.mark.parametrize("excitation", ["nonselective", "slab"])
def test_a_cycle_repeats_one_balanced_repetition_from_its_first_excitation(excitation):
    seq = main_3d(excitation=excitation, ry=2, n_acs_y=4, n_acs_z=2)
    rows = excitations(seq)

    assert seq.repetition() == (3, 1)
    assert seq.get_block(1).rf is not None
    assert np.diff([time for time, _, _ in rows]) == pytest.approx(recorded(seq, "TR"))


def test_every_repetition_returns_its_gradient_moments_to_zero():
    seq = main_3d(ry=2, n_acs_y=4, n_acs_z=2)
    waves, pulses = seq.waveforms_and_times()[:2]
    edges = np.asarray(pulses)[0]
    for channel in waves:
        t, g = np.asarray(channel)
        if t.size == 0:
            continue
        grid = np.union1d(t, edges)
        amplitude = np.interp(grid, t, g, left=0.0, right=0.0)
        area = np.concatenate(
            [[0.0], np.cumsum(np.diff(grid) * (amplitude[1:] + amplitude[:-1]) / 2)]
        )
        assert np.diff(np.interp(edges, grid, area)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("n_cycles", [1, 2, 4])
def test_each_cycle_steps_its_phase_by_its_own_increment(n_cycles):
    for k, seq in enumerate(chain_3d(n_phase_cycles=n_cycles)[1::2]):
        rows = excitations(seq)
        increment = np.pi + 2 * np.pi * k / n_cycles
        adc = [block.adc.phase_offset for block in blocks(seq) if block.adc is not None]
        (cycle,) = labels(seq, "SET")

        assert same_phase(
            [p for _, p, _ in rows], increment * np.arange(1, len(rows) + 1)
        )
        assert same_phase(adc, [p for _, p, _ in rows])
        assert recorded(seq, "PhaseIncrement") == pytest.approx(np.rad2deg(increment))
        assert set(cycle) == {k}


def test_a_catalyst_is_a_half_flip_half_a_repetition_before_its_cycle():
    catalyst, cycle = chain_3d()
    (half,) = excitations(catalyst)
    first = excitations(cycle)[0]
    increment = np.deg2rad(recorded(cycle, "PhaseIncrement"))

    assert half[2] == pytest.approx(0.5 * first[2])
    assert same_phase(half[1], first[1] - increment)
    lead = catalyst.duration()[0] - half[0]
    assert lead + first[0] == pytest.approx(0.5 * recorded(cycle, "TR"), abs=1e-9)


def test_every_view_is_read_once_inside_the_ellipse_back_and_forth():
    n_y, n_z = SMALL_3D["n_y"], 8
    seq = main_3d(ry=2, rz=2, n_z=n_z, n_acs_y=4, n_acs_z=2)
    lin, par, ima = labels(seq, "LIN", "PAR", "IMA")
    views = list(zip(lin, par, strict=True))
    calibrating, imaging = pp.make_cartesian_plane_sampling(
        (n_y, n_z), (2, 2), (4, 2), elliptical=True
    )

    assert views == sorted({*calibrating, *imaging})
    assert len(set(views)) == len(views)
    assert all(
        ((y - 8) / 16) ** 2 + ((z - 4) / 8) ** 2 <= 0.25 or (y, z) in calibrating
        for y, z in views
    )
    assert list(ima) == [int(view in calibrating) for view in views]
    steps = [abs(b[1] - a[1]) for a, b in pairwise(views)]
    assert max(steps) <= 4


@pytest.mark.parametrize(
    "prescription",
    [{"excitation": "spsp"}, {"tr": 1e-4}, {"n_phase_cycles": 0}, {"caipi_shift": 2}],
    ids=str,
)
def test_a_3d_prescription_that_cannot_be_played_is_refused(prescription):
    with pytest.raises(ValueError):
        chain_3d(**prescription)


@pytest.mark.parametrize(
    ("module", "flag", "help_text"),
    [
        (bssfp2d, "--gating", "No gating; each segment cycled over one heartbeat"),
        (bssfp3d, "--n-phase-cycles", "Trains acquired, each with its own RF phase"),
    ],
)
def test_a_bssfp_flag_is_named_and_described_by_its_docstring(
    capsys, module, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module.main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())

    assert flag in printed
    assert help_text in printed

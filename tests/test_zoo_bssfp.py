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


def app2d(**kwargs):
    return bssfp2d.Bssfp2DApp(pp.Opts(), **{**SMALL_2D, **kwargs})


def app3d(**kwargs):
    return bssfp3d.Bssfp3DApp(pp.Opts(), **{**SMALL_3D, **kwargs})


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
    built = app2d(n_slices=3, slice_spacing=1e-3, **prescription)
    seq = built.design()
    size, start = seq._detect_tr()

    assert seq.check_timing()[0]
    assert start == 1
    assert 3 * size == len(seq.block_events)


def test_once_marks_each_slice_half_flip_and_closing_rewind():
    seq = app2d(n_slices=2).design()
    (once,) = labels(seq, "ONCE", evolution="blocks")

    for chunk in np.split(once, 2):
        assert chunk[0] == 1
        assert chunk[-1] == 2
        assert set(chunk[1:-1]) == {0}


def test_the_phase_alternates_after_an_opposite_half_flip():
    built = app2d(n_slices=2, slice_spacing=1e-3)
    seq = built.design()
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
        assert offsets[start + 1] - offsets[start] == pytest.approx(
            0.5 * built.ro.tr, abs=1e-9
        )


@pytest.mark.parametrize("ry", [1, 2])
def test_every_line_of_every_slice_is_read_once_when_ungated(ry):
    built = app2d(n_slices=2, ry=ry, n_acs_y=4)
    lin, slc, ima = labels(built.design(), "LIN", "SLC", "IMA")

    assert list(zip(lin, slc, strict=True)) == [
        (line, s) for s in range(2) for line in built.lines
    ]
    assert list(ima) == [int(line in built.calibration) for line in lin]


def test_a_retrospective_heartbeat_cycles_its_segment():
    built = app2d(gating="retrospective", **CINE)
    lin, seg, phs = labels(built.design(), "LIN", "SEG", "PHS")
    cycles = max(1, round(0.2 / (4 * built.ro.tr)))
    expected = [
        (line, s, c)
        for s, start in enumerate(range(0, 16, 4))
        for c in range(cycles)
        for line in built.lines[start : start + 4]
    ]

    assert cycles > 1
    assert list(zip(lin, seg, phs, strict=True)) == expected
    assert not any(getattr(block, "trig", None) for block in blocks(built.design()))


def test_a_prospective_heartbeat_waits_for_its_trigger_then_reads_every_phase():
    built = app2d(gating="prospective", n_phases=3, trigger_delay=5e-3, **CINE)
    seq = built.design()
    lin, seg, phs = labels(seq, "LIN", "SEG", "PHS")
    triggers = [block.trig for block in blocks(seq) if getattr(block, "trig", None)]
    expected = [
        (line, s, f)
        for s, start in enumerate(range(0, 16, 4))
        for f in range(3)
        for line in built.lines[start : start + 4]
    ]

    assert list(zip(lin, seg, phs, strict=True)) == expected
    assert len(triggers) == 4
    assert all(t[0].channel == "physio1" for t in triggers)
    assert all(t[0].duration == pytest.approx(5e-3) for t in triggers)
    assert seq.definitions["CardiacPhases"] == pytest.approx([3])


@pytest.mark.parametrize("n_dummy", [0, 3])
def test_the_first_trigger_of_a_slice_precedes_its_half_flip(n_dummy):
    built = app2d(gating="prospective", n_phases=2, n_slices=2, n_dummy=n_dummy, **CINE)
    seq = built.design()
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
    built = app2d(gating="prospective", n_phases=1, views_per_segment=16, n_slices=2)
    triggers = [b for b in blocks(built.design()) if getattr(b, "trig", None)]

    assert built.n_segments == 1
    assert len(triggers) == 2


def test_every_trigger_falls_between_two_balanced_repetitions():
    seq = app2d(gating="prospective", n_phases=2, **CINE).design()
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
        app2d(**prescription)


# -- 3D ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prescription", "files"),
    [
        ({}, [("scan.seq", "bssfp_3d_catalyst"), ("scan_main.seq", "bssfp_3d")]),
        (
            {"n_phase_cycles": 2},
            [
                ("scan.seq", "bssfp_3d_catalyst"),
                ("scan_cycle_0.seq", "bssfp_3d"),
                ("scan_catalyst_1.seq", "bssfp_3d_catalyst"),
                ("scan_main.seq", "bssfp_3d"),
            ],
        ),
    ],
    ids=["one cycle", "two cycles"],
)
def test_the_chain_alternates_catalysts_and_cycles(tmp_path, prescription, files):
    built = app3d(**prescription)
    paths = built.write(tmp_path / "scan.seq")

    assert [Path(p).name for p in paths] == [name for name, _ in files]
    for i, path in enumerate(paths):
        seq = pp.Sequence(system=built.system)
        seq.read(path)
        following = files[i + 1][0] if i + 1 < len(files) else None
        assert seq.definitions["Name"] == files[i][1]
        assert seq.definitions.get("NextSequence") == following
        assert seq.check_timing()[0]
        assert seq._detect_tr()[1] == 1


@pytest.mark.parametrize("excitation", ["nonselective", "slab"])
def test_a_cycle_repeats_one_balanced_repetition_from_its_first_excitation(excitation):
    built = app3d(excitation=excitation, ry=2, n_acs_y=4, n_acs_z=2)
    seq = built.design()
    rows = excitations(seq)

    assert seq._detect_tr() == (3, 1)
    assert seq.get_block(1).rf is not None
    assert np.diff([time for time, _, _ in rows]) == pytest.approx(built.ro.tr)


def test_every_repetition_returns_its_gradient_moments_to_zero():
    seq = app3d(ry=2, n_acs_y=4, n_acs_z=2).design()
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
    built = app3d(n_phase_cycles=n_cycles)
    for k in range(n_cycles):
        seq = built.design(None if k == n_cycles - 1 else f"cycle_{k}")
        rows = excitations(seq)
        increment = np.pi + 2 * np.pi * k / n_cycles
        adc = [block.adc.phase_offset for block in blocks(seq) if block.adc is not None]
        (cycle,) = labels(seq, "SET")

        assert same_phase(
            [p for _, p, _ in rows], increment * np.arange(1, len(rows) + 1)
        )
        assert same_phase(adc, [p for _, p, _ in rows])
        assert set(cycle) == {k}


def test_a_catalyst_is_a_half_flip_half_a_repetition_before_its_cycle():
    built = app3d()
    catalyst = built.design("catalyst_0")
    cycle = built.design()
    (half,) = excitations(catalyst)
    first = excitations(cycle)[0]

    assert half[2] == pytest.approx(0.5 * first[2])
    assert same_phase(half[1], first[1] - built.increments[0])
    lead = catalyst.duration()[0] - half[0]
    assert lead + first[0] == pytest.approx(0.5 * built.ro.tr, abs=1e-9)


def test_every_view_is_read_once_inside_the_ellipse_back_and_forth():
    built = app3d(ry=2, rz=2, n_z=8, n_acs_y=4, n_acs_z=2)
    lin, par, ima = labels(built.design(), "LIN", "PAR", "IMA")
    views = list(zip(lin, par, strict=True))

    assert views == built.views
    assert len(set(views)) == len(views)
    assert all(
        ((y - 8) / 16) ** 2 + ((z - 4) / 8) ** 2 <= 0.25 or (y, z) in built.calibration
        for y, z in views
    )
    assert list(ima) == [int(view in built.calibration) for view in views]
    steps = [abs(b[1] - a[1]) for a, b in pairwise(views)]
    assert max(steps) <= 4


@pytest.mark.parametrize(
    "prescription",
    [{"excitation": "spsp"}, {"tr": 1e-4}, {"n_phase_cycles": 0}, {"caipi_shift": 2}],
    ids=str,
)
def test_a_3d_prescription_that_cannot_be_played_is_refused(prescription):
    with pytest.raises(ValueError):
        app3d(**prescription)


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

"""Sequence examples, recorded prescriptions and generated command-line interfaces."""

import inspect
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from zoo import (
    FUNCTIONS,
    SMALL,
    application,
    chain,
    is_application,
    legacy_application,
    packets,
)

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._prescription import documented


def test_every_zoo_entry_has_a_small_prescription():
    assert set(SMALL) == set(sequences.ZOO)


@pytest.mark.parametrize("name", sequences.ZOO)
def test_a_zoo_entry_is_callable_as_the_sequence_it_builds(name):
    """A script is its main, docstring and signature included."""
    script = getattr(sequences, name)

    assert callable(script)
    assert script.__doc__ == script.main.__doc__
    assert "system" in script.__signature__.parameters


@pytest.mark.parametrize("name", sequences.ZOO)
def test_a_zoo_entry_takes_its_dummies_and_names_the_axis_of_its_calibration(name):
    parameters = getattr(sequences, name).__signature__.parameters
    calibration = [p for p in parameters if p.startswith("n_acs")]

    # The 3D bSSFP starts its steady state with a chained half flip instead.
    assert ("n_dummy" in parameters) != (name == "bssfp3D_sequence")
    assert all(p in ("n_acs_y", "n_acs_z") for p in calibration)


@pytest.mark.parametrize("name", sequences.ZOO)
def test_a_zoo_entry_accepts_its_default_protocol(name):
    """The design refuses a TE, a spacing or a TR it cannot meet."""
    if is_application(name):
        application(name)(pp.Opts())
    else:
        getattr(sequences, name).main(pp.Opts())


@pytest.mark.parametrize("name", sequences.ZOO)
def test_every_sequence_a_zoo_entry_builds_passes_its_timing_check(name):
    built = chain(getattr(sequences, name)(**SMALL[name]))

    assert all(isinstance(seq, pp.Sequence) for seq in built)
    for seq in built:
        is_ok, errors = seq.check_timing()
        assert is_ok, errors


@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_function_takes_the_system_first_and_the_protocol_as_keywords(name):
    first, *protocol = inspect.signature(
        getattr(sequences, name).main
    ).parameters.values()

    assert (first.name, first.default) == ("system", None)
    assert {parameter.kind for parameter in protocol} == {
        inspect.Parameter.KEYWORD_ONLY
    }


@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_function_states_the_limits_it_holds_the_system_to(name):
    module = getattr(sequences, name)
    _, entry = documented(inspect.getdoc(module.main))["system"]

    assert f"({module.MAX_GRAD:g} mT/m)" in entry
    assert f"({module.MAX_SLEW:g} T/m/s)" in entry


@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_function_lowers_the_system_to_its_limits_and_never_raises_it(name):
    module = getattr(sequences, name)

    def hardware(scale):
        return pp.Opts(
            max_grad=scale * module.MAX_GRAD,
            grad_unit="mT/m",
            max_slew=scale * module.MAX_SLEW,
            slew_unit="T/m/s",
        )

    above = chain(module.main(hardware(2), **SMALL[name]))[-1].system
    below = chain(module.main(hardware(0.5), **SMALL[name]))[-1].system

    assert above.max_grad == pytest.approx(hardware(1).max_grad)
    assert above.max_slew == pytest.approx(hardware(1).max_slew)
    assert below.max_grad == pytest.approx(hardware(0.5).max_grad)
    assert below.max_slew == pytest.approx(hardware(0.5).max_slew)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_function_called_twice_writes_the_same_files(tmp_path, name):
    """Nothing a call builds outlives it: label state, caches, module constants."""
    module = getattr(sequences, name)
    written = []
    for run in ("first", "second"):
        (tmp_path / run).mkdir()
        paths = sequences.write(tmp_path / run / "scan.seq", module.main(**SMALL[name]))
        written.append([(Path(path).name, Path(path).read_bytes()) for path in paths])

    assert written[0] == written[1]


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


# -- what the 2D gradient echo is ------------------------------------------


def gre(**kwargs):
    return sequences.gre2D_sequence(**{**SMALL["gre2D_sequence"], **kwargs})


def gre_app(**kwargs):
    """The 2D gradient echo as a SequenceApp, without dummies unless asked for."""
    app = legacy_application("gre2D_sequence")
    return app(pp.Opts(), **{**SMALL["gre2D_sequence"], "n_dummy": 0, **kwargs})


def test_every_line_is_one_repetition_of_the_same_blocks():
    seq = gre(n_y=16)

    assert len(seq.block_events) % 16 == 0


def test_the_prescription_asked_for_is_the_one_written_down():
    seq = gre(n_x=64, n_y=32, n_slices=3, slice_thickness=4e-3, fov_x=0.2, fov_y=0.2)

    assert seq.definitions["Matrix"] == [64.0, 32.0, 3.0]
    assert seq.definitions["FOV"] == pytest.approx([0.2, 0.2, 12e-3])
    assert seq.definitions["Name"] == "gre_2d"
    assert len(seq.definitions["SlicePositions"]) == 3


def test_the_slices_of_a_packet_are_not_neighbours():
    """A TR too short for every slice deals them into packets, spread out."""
    played = packets(gre(n_slices=8, tr=40e-3, n_dummy=0))

    assert len(played) > 1
    for packet in played:
        # Every slice of one packet is a whole packet count away from the next,
        # so no two neighbours in the slab are excited in the same packet.
        assert all(b - a >= len(played) for a, b in pairwise(sorted(packet)))


def test_the_even_slices_of_a_packet_are_excited_before_the_odd_ones():
    played = packets(gre(n_slices=10, tr=None, n_dummy=0))

    assert played == [[0, 2, 4, 6, 8, 1, 3, 5, 7, 9]]


@pytest.mark.parametrize(
    ("n_x", "n_slices"),
    [(64, 120), (256, 120), (256, 30), (256, 7)],
    ids=["even packets", "odd packet", "even", "one packet"],
)
def test_the_scan_repeats_from_its_first_block_whatever_the_slices_divide_into(
    n_x, n_slices
):
    """A packet that holds one slice more is a longer wait, not a different shot."""
    seq = gre(n_x=n_x, n_y=32, n_slices=n_slices, n_acs_y=8, readout_oversampling=1.0)

    _size, start = seq.repetition()

    assert start == 1


def test_every_slice_is_excited_at_the_repetition_time_asked_for():
    """Including the odd packet, which holds a slice more and waits less."""
    lines, tr = 8, 0.25
    seq = gre(
        n_x=256,
        n_y=lines,
        n_slices=120,
        tr=tr,
        readout_oversampling=1.0,
        n_dummy=0,
    )
    played = packets(seq)
    excited = np.asarray(seq.rf_times()[0])

    assert len({len(packet) for packet in played}) == 2  # the case worth asking

    at = 0
    for packet in played:
        # A slice's repetition time is the gap between its own excitations.
        spacing = np.diff(excited[at : at + len(packet) * lines][:: len(packet)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(packet) * lines


def test_the_repetition_time_written_is_the_spacing_of_one_slices_excitations():
    seq = gre(n_slices=3, tr=0.05, n_dummy=0)
    excited = np.asarray(seq.rf_times()[0])

    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(0.05)
    assert excited[3] - excited[0] == pytest.approx(0.05)


def test_the_slice_thickness_written_is_the_one_excited_and_the_gap_keeps_the_centres():
    """The thickness is the pulse's bandwidth over its selection gradient."""
    seq = gre(n_slices=3, slice_thickness=4e-3, slice_spacing=1e-3, n_dummy=0)
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    excitations = [block for block in blocks if block.rf is not None]
    selection = abs(excitations[0].gz.amplitude)
    centres = sorted({round(b.rf.freq_offset / selection, 9) for b in excitations})
    thickness = np.atleast_1d(seq.definitions["SliceThickness"])[0]
    gap = np.atleast_1d(seq.definitions["SliceGap"])[0]

    assert thickness == pytest.approx(
        pp.calc_rf_bandwidth(excitations[0].rf) / selection
    )
    assert thickness == pytest.approx(4e-3)
    assert np.diff(centres) == pytest.approx(thickness + gap)


def test_a_repetition_that_holds_whole_shots_takes_that_many_slices_a_packet():
    """With one slice and the shortest TR, the TR written is one shot."""
    shot = np.atleast_1d(gre(n_slices=1, tr=None, n_dummy=0).definitions["TR"])[0]
    played = packets(gre(n_slices=4, tr=2 * shot, n_dummy=0))

    assert [len(packet) for packet in played] == [2, 2]


def test_a_shorter_echo_than_the_readout_admits_is_refused():
    with pytest.raises(ValueError):
        gre(te=1e-6)


def test_undersampling_acquires_fewer_lines_than_it_encodes():
    full = gre(n_y=32, ry=1)
    half = gre(n_y=32, ry=2)

    assert len(half.block_events) < len(full.block_events)


@pytest.mark.parametrize("n_y", [32, 33])
@pytest.mark.parametrize("ry", [2, 3, 4, 5])
def test_undersampling_always_acquires_the_centre_line(n_y, ry):
    (lines,) = adc_labels(gre(n_y=n_y, ry=ry, n_acs_y=0, n_dummy=0), "LIN")

    assert n_y // 2 in lines
    assert all((line - n_y // 2) % ry == 0 for line in lines)


def test_partial_fourier_drops_lines_before_the_centre_only():
    (lines,) = adc_labels(gre(n_y=32, partial_fourier_y=0.75, n_dummy=0), "LIN")

    assert list(lines) == list(range(8, 32))


@pytest.mark.parametrize("name", ["partial_fourier_x", "partial_fourier_y"])
@pytest.mark.parametrize("fraction", [0.7, 1.01])
def test_a_partial_fourier_fraction_outside_its_range_is_refused(name, fraction):
    with pytest.raises(ValueError, match=name):
        gre(**{name: fraction})


def test_every_slice_is_rf_spoiled_by_its_own_excitation_count():
    """Consecutive excitations of one slice step their phase by a growing 117 degrees."""
    seq = gre(n_slices=3, n_y=8, tr=None, n_dummy=0)
    phases = {}
    for index in range(1, len(seq.block_events) + 1):
        rf = getattr(seq.get_block(index), "rf", None)
        if rf is not None:
            # The slice offset's phase ramp is not part of the spoiling.
            spoiling = rf.phase_offset + 2 * np.pi * rf.freq_offset * rf.center
            phases.setdefault(round(rf.freq_offset, 3), []).append(spoiling)

    assert len(phases) == 3
    for slice_phases in phases.values():
        steps = np.diff(np.diff(slice_phases))
        wrapped = np.angle(np.exp(1j * (steps - np.deg2rad(117.0))))
        assert wrapped == pytest.approx(0.0, abs=1e-9)


def test_the_calibration_block_leads_the_scan():
    """A reconstruction calibrates while the rest of the scan is arriving."""
    lin, ima = adc_labels(gre(n_y=32, ry=2, n_acs_y=8, n_dummy=0), "LIN", "IMA")

    assert list(ima) == [1] * 8 + [0] * (len(ima) - 8)
    assert list(lin[:8]) == sorted(lin[ima == 1])


def test_a_fully_sampled_scan_has_no_calibration_block():
    lin, ima = adc_labels(gre(n_y=32, ry=1, n_acs_y=8, n_dummy=0), "LIN", "IMA")

    assert not ima.any()
    assert list(lin) == list(range(32))


def test_each_acquisition_carries_the_line_and_slice_it_encodes():
    seq = gre(n_dummy=2, n_y=16, n_slices=3, ry=2, n_acs_y=4)
    lin, slc, ima, seg = adc_labels(seq, "LIN", "SLC", "IMA", "SEG")
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        16, 2, 4, partial_fourier=1.0
    )
    # One packet, whose even slices are excited before its odd ones.
    expected = [(line, s) for line in [*calibrating, *imaging] for s in (0, 2, 1)]

    assert list(zip(lin, slc, strict=True)) == expected
    assert list(ima) == [int(line in calibrating) for line, _ in expected]
    assert list(seg) == [1 - int(line in calibrating) for line, _ in expected]


# -- the application contract ----------------------------------------------


def test_an_application_states_its_gradient_limits():
    class Unlimited(sequences.SequenceApp):
        def init_sequence(self):
            pass

        def kernel(self):
            pass

        def loop(self):
            pass

    with pytest.raises(TypeError, match="MAX_GRAD or MAX_SLEW"):
        Unlimited()


def test_a_subclass_changes_a_setting_and_nothing_else():
    class Gentle(legacy_application("gre2D_sequence")):
        MAX_SLEW = 100.0

    assert Gentle(pp.Opts(), **SMALL["gre2D_sequence"]).system.max_slew < (
        gre_app().system.max_slew
    )


def test_the_protocol_is_the_prescription_with_its_defaults():
    protocol = legacy_application("gre2D_sequence").protocol()

    assert protocol["n_y"] == 128
    assert protocol["tr"] == 250e-3


def test_a_repeated_step_is_an_inc_and_any_other_change_a_set():
    app = gre_app()
    kinds = [
        [(e.type, e.value) for e in app.labels(LIN=line)] for line in (0, 2, 4, 6, 3)
    ]

    assert kinds == [
        [("labelset", 0)],
        [("labelset", 2)],
        [("labelinc", 2)],
        [("labelinc", 2)],
        [("labelset", 3)],
    ]
    assert app.labels(LIN=3) == []


def test_a_change_of_once_makes_the_next_change_of_every_label_a_set():
    """The blocks either side of a ONCE change are not always played in turn."""
    app = gre_app()
    for line in (0, 2, 4):
        app.labels(LIN=line, ONCE=1)
    events = app.labels(LIN=6, ONCE=0)

    assert [(e.label, e.type) for e in events] == [
        ("LIN", "labelset"),
        ("ONCE", "labelset"),
    ]


def test_a_label_set_before_a_change_of_once_is_written_again_after_it():
    """The dummies' slice is not there when a repeat skips the dummies."""
    app = gre_app()
    app.labels(SLC=2, ONCE=1)
    events = app.labels(SLC=2, LIN=0, ONCE=0)

    assert {(e.label, e.type, e.value) for e in events} == {
        ("SLC", "labelset", 2),
        ("LIN", "labelset", 0),
        ("ONCE", "labelset", 0),
    }


def test_a_prescan_is_written_first_and_names_the_main_sequence_next(tmp_path):
    class Prescanned(legacy_application("gre2D_sequence")):
        def prescans(self):
            return {"calibration": lambda: self.kernel(0, 8, 0.0, self.raster)}

    app = Prescanned(pp.Opts(), **SMALL["gre2D_sequence"])
    paths = app.write(tmp_path / "scan.seq")
    first, main = pp.Sequence(), pp.Sequence()
    first.read(paths[0])
    main.read(paths[1])

    assert [Path(p).name for p in paths] == ["scan.seq", "scan_main.seq"]
    assert first.definitions["NextSequence"] == "scan_main.seq"
    assert "NextSequence" not in main.definitions
    assert len(first.block_events) < len(main.block_events)


def test_without_prescans_the_scan_is_one_file(tmp_path):
    paths = gre_app().write(tmp_path / "scan.seq")

    assert paths == [str(tmp_path / "scan.seq")]


def test_one_call_plays_one_repetition():
    app = gre_app()
    app(0, 8, 0.0, app.raster)

    assert len(app.seq.rf_times()[0]) == 1
    assert app.seq.evaluate_labels(evolution="adc")["LIN"] == 8


def test_designing_twice_gives_the_same_scan():
    app = gre_app()

    assert len(app.design().block_events) == len(app.design().block_events)


# -- the command line ------------------------------------------------------


def test_the_command_line_writes_what_the_call_builds(tmp_path):
    path = tmp_path / "gre.seq"

    status = cli.run(
        sequences.gre2D_sequence.main,
        [
            "-o",
            str(path),
            "--n-x",
            "32",
            "--n-y",
            "16",
            "--n-acs",
            "0",
        ],
    )

    assert status == 0
    assert path.read_text().startswith("# Pulseq sequence file")


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        ("gre2D_sequence", "--flip-angle-deg", "Excitation flip angle (degrees)."),
        ("fse3D_sequence", "--flip-modulation", "Constant refocusing angles, or"),
    ],
)
def test_a_flag_is_named_and_described_by_the_function_it_runs(
    capsys, name, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(getattr(sequences, name).main, ["--help"])

    printed = capsys.readouterr().out

    assert flag in printed
    assert help_text in printed


def test_a_flag_is_described_by_the_first_sentence_whatever_lines_it_spans(capsys):
    def main(fov_x: float = 0.2, fov_y: float = 0.2, flyback: bool = True):
        """Build nothing.

        Parameters
        ----------
        fov_x, fov_y : float, optional
            Field of view along the readout and the phase encode, in metres,
            as prescribed. Ignored.
        flyback : bool, optional
            Monopolar echo train; off, a
            bipolar one. Ignored.
        """

    with pytest.raises(SystemExit):
        cli.run(main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())
    fov = "Field of view along the readout and the phase encode, in metres, as prescribed."

    assert f"--fov-x FOV_X {fov} --fov-y FOV_Y {fov}" in printed
    assert printed.endswith("--no-flyback Monopolar echo train; off, a bipolar one.")


def test_running_the_module_as_a_script_writes_a_sequence(tmp_path):
    path = tmp_path / "gre.seq"

    # The interpreter running these tests, and a path pytest made.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pypulseqpp.sequences.sequence.gre2D_sequence",
            "-o",
            str(path),
            "--n-x",
            "32",
            "--n-y",
            "16",
            "--n-acs",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert path.exists()

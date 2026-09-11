"""Sequence examples, recorded prescriptions and generated command-line interfaces."""

import subprocess
import sys
from itertools import pairwise

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: A prescription small enough to build in a moment, per zoo entry.
SMALL = {
    "gre2D_sequence": {"n_x": 32, "n_y": 16, "n_slices": 1, "n_acs": 0, "n_dummy": 0},
    "gre3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "n_acs": 0,
        "n_acs_z": 0,
        "n_dummy": 0,
    },
    "gre_multiecho2D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_echoes": 3,
        "n_acs": 0,
        "n_dummy": 0,
    },
    "gre_multiecho3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "n_echoes": 3,
        "n_acs": 0,
        "n_acs_z": 0,
        "n_dummy": 0,
    },
    "gre_stack_of_stars3D_sequence": {"n_x": 32, "n_z": 4, "n_spokes": 5, "n_dummy": 0},
    "gre_stack_of_spirals3D_sequence": {"n_x": 32, "n_z": 4, "n_arms": 4, "n_dummy": 0},
    "zte3D_sequence": {"n_x": 32, "n_views": 24, "n_shots": 2, "n_dummy": 0},
    "gre_radial2D_sequence": {"n_x": 32, "n_spokes": 8, "n_dummy": 0},
    "gre_spiral2D_sequence": {"n_x": 32, "n_arms": 4, "n_dummy": 0},
    "se_propeller2D_sequence": {
        "n_x": 32,
        "blade_width": 8,
        "n_blades": 4,
        "te": None,
        "tr": None,
    },
    "epi2D_sequence": {"n_x": 32, "n_y": 16, "n_dummy": 0},
    "epi3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 4, "n_dummy": 0},
    "se2D_sequence": {"n_x": 32, "n_y": 16, "n_slices": 1, "n_acs": 0, "tr": None},
    "se3D_sequence": {
        "n_x": 32,
        "n_y": 8,
        "n_z": 4,
        "n_acs": 0,
        "n_acs_z": 0,
        "tr": None,
    },
    "fse2D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_slices": 1,
        "etl": 4,
        "n_acs": 0,
        "te": None,
        "tr": None,
    },
    "mprage3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "views_per_segment": 16,
        "ti": 100e-3,
        "tr_outer": 300e-3,
        "n_acs": 0,
        "n_acs_z": 0,
        "n_dummy": 0,
    },
    "mprage_stack_of_spirals3D_sequence": {
        "n_x": 32,
        "n_z": 4,
        "n_arms": 4,
        "ti": 100e-3,
        "tr_outer": 300e-3,
        "n_dummy": 0,
    },
    "bssfp2D_sequence": {
        "n_x": 64,
        "n_y": 16,
        "readout_bandwidth_hz": 50e3,
        "n_acs": 0,
        "n_dummy": 0,
    },
    "fse3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "etl": 4,
        "n_acs": 0,
        "n_acs_z": 0,
        "te": None,
        "tr": None,
    },
}


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
def test_a_zoo_entry_builds_a_sequence_that_passes_its_timing_check(name):
    seq = getattr(sequences, name)(**SMALL[name])

    assert isinstance(seq, pp.Sequence)
    is_ok, errors = seq.check_timing()
    assert is_ok, errors


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


# -- what the 2D gradient echo is ------------------------------------------


def gre(**kwargs):
    return sequences.gre2D_sequence(**{**SMALL["gre2D_sequence"], **kwargs})


def gre_app(**kwargs):
    return sequences.gre2D_sequence.Gre2DApp(
        pp.Opts(), **{**SMALL["gre2D_sequence"], **kwargs}
    )


def test_every_line_is_one_repetition_of_the_same_blocks():
    seq = gre(n_y=16)

    assert len(seq.block_events) % 16 == 0


def test_the_prescription_asked_for_is_the_one_written_down():
    seq = gre(n_x=64, n_y=32, n_slices=3, slice_thickness=4e-3, fov=0.2)

    assert seq.definitions["Matrix"] == [64.0, 32.0, 3.0]
    assert seq.definitions["FOV"] == pytest.approx([0.2, 0.2, 12e-3])
    assert seq.definitions["Name"] == "gre_2d"
    assert len(seq.definitions["SlicePositions"]) == 3


def test_the_slices_of_a_pass_are_not_neighbours():
    """A TR too short for every slice deals them into passes, spread out."""
    app = gre_app(n_slices=8, tr=40e-3)

    assert len(app.passes) > 1
    for group in app.passes:
        # Every slice of one pass is a whole pass count away from the next, so
        # no two neighbours in the slab are excited in the same pass.
        assert all(b - a >= len(app.passes) for a, b in pairwise(sorted(group)))


@pytest.mark.parametrize(
    ("n_x", "n_slices"),
    [(64, 120), (256, 120), (256, 30), (256, 7)],
    ids=["even passes", "odd pass", "even", "one pass"],
)
def test_the_scan_repeats_from_its_first_block_whatever_the_slices_divide_into(
    n_x, n_slices
):
    """A pass that holds one slice more is a longer wait, not a different shot."""
    seq = gre(n_x=n_x, n_y=32, n_slices=n_slices, n_acs=8)

    _size, start = seq._detect_tr()

    assert start == 1


def test_every_slice_is_excited_at_the_repetition_time_asked_for():
    """Including the odd pass, which holds a slice more and waits less."""
    lines, tr = 8, 0.25
    app = gre_app(n_x=256, n_y=lines, n_slices=120, tr=tr)
    excited = np.asarray(app.design().rf_times()[0])

    assert len({len(group) for group in app.passes}) == 2  # the case worth asking

    at = 0
    for group in app.passes:
        # A slice's repetition time is the gap between its own excitations.
        spacing = np.diff(excited[at : at + len(group) * lines][:: len(group)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(group) * lines


def test_the_repetition_time_written_is_the_spacing_of_one_slices_excitations():
    app = gre_app(n_slices=3, tr=0.05)
    seq = app.design()
    excited = np.asarray(seq.rf_times()[0])

    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(0.05)
    assert excited[3] - excited[0] == pytest.approx(0.05)


def test_a_repetition_that_holds_whole_shots_takes_that_many_slices_a_pass():
    shot = gre_app().ro.duration + pp.Opts().block_duration_raster
    app = gre_app(n_slices=4, tr=2 * shot)

    assert [len(group) for group in app.passes] == [2, 2]


def test_a_shorter_echo_than_the_readout_admits_is_refused():
    with pytest.raises(ValueError):
        gre(te=1e-6)


def test_undersampling_acquires_fewer_lines_than_it_encodes():
    full = gre(n_y=32, acceleration=1)
    half = gre(n_y=32, acceleration=2)

    assert len(half.block_events) < len(full.block_events)


def test_the_calibration_block_leads_the_scan():
    """A reconstruction calibrates while the rest of the scan is arriving."""
    app = gre_app(n_y=32, acceleration=2, n_acs=8)

    assert list(app.lines[:8]) == sorted(app.calibration)


def test_each_acquisition_carries_the_line_and_slice_it_encodes():
    app = gre_app(n_y=16, n_slices=3, acceleration=2, n_acs=4, n_dummy=2)
    lin, slc, ima, seg = adc_labels(app.design(), "LIN", "SLC", "IMA", "SEG")

    expected = [(line, s) for group in app.passes for line in app.lines for s in group]
    assert list(zip(lin, slc, strict=True)) == expected
    assert list(ima) == [int(line in app.calibration) for line, _ in expected]
    assert list(seg) == [1 - int(line in app.calibration) for line, _ in expected]


# -- what the 3D fast spin echo is -----------------------------------------


def fse(**kwargs):
    return sequences.fse3D_sequence(**{**SMALL["fse3D_sequence"], **kwargs})


def fse_app(**kwargs):
    return sequences.fse3D_sequence.Fse3DApp(
        pp.Opts(), **{**SMALL["fse3D_sequence"], **kwargs}
    )


@pytest.mark.parametrize("ordering", sequences.fse3D_sequence.ORDERINGS)
def test_every_sampled_view_is_acquired_once_at_its_place_in_the_train(ordering):
    app = fse_app(ordering=ordering, acceleration=2, n_acs=4, n_acs_z=2)
    lin, par, eco = adc_labels(app.design(), "LIN", "PAR", "ECO")

    views = [v for train in app.trains for v in train if v is not None]
    echoes = [e for train in app.trains for e, v in enumerate(train) if v is not None]

    assert list(zip(lin, par, strict=True)) == views
    assert len(set(views)) == len(views)
    assert list(eco) == echoes


def test_the_effective_echo_is_the_one_asked_for():
    app = fse_app(etl=8, te=40e-3, ordering="linear")
    written = np.atleast_1d(app.design().definitions["TE"])[0]

    assert written == pytest.approx(app.fse.echo_times[app.n_center])
    assert abs(written - 40e-3) <= app.fse.esp / 2


def test_a_variable_train_starts_and_ends_at_its_largest_flip():
    flips = sequences.fse3D_sequence.traps_flip_schedule(16, 8)

    assert flips[0] == flips[-1] == 160.0
    assert flips.min() == 60.0
    assert flips[8] == 100.0


def test_the_wave_free_calibration_trains_lead_and_are_marked_reference():
    app = fse_app(wave="both", wave_cycles=2, n_acs=4, n_acs_z=2)
    seq = app.design()
    ref, seg = adc_labels(seq, "REF", "SEG")
    n_reference = len(app.calibration_views)

    assert seq.check_timing()[0]
    assert list(ref[:n_reference]) == [1] * n_reference
    assert set(ref[n_reference:]) == {0}
    assert set(seg[n_reference:]) == {1}


def test_navigators_fit_in_the_repetition_asked_for():
    app = fse_app(navigator=True, tr=1.0)
    seq = app.design()

    assert app.n_navigators > 0
    assert app.repetition_time == pytest.approx(1.0)
    assert seq.check_timing()[0]


def test_a_repetition_shorter_than_the_train_is_refused():
    with pytest.raises(ValueError, match="shorter than one train"):
        fse(tr=1e-3)


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
    class Gentle(sequences.gre2D_sequence.Gre2DApp):
        MAX_SLEW = 100.0

    assert Gentle(pp.Opts(), **SMALL["gre2D_sequence"]).system.max_slew < (
        gre_app().system.max_slew
    )


def test_the_protocol_is_the_prescription_with_its_defaults():
    protocol = sequences.gre2D_sequence.Gre2DApp.protocol()

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
    class Prescanned(sequences.gre2D_sequence.Gre2DApp):
        def prescans(self):
            return {"calibration": lambda: self.kernel(0, 8, 0.0, self.raster)}

    app = Prescanned(pp.Opts(), **SMALL["gre2D_sequence"])
    paths = app.write(tmp_path / "scan.seq")
    first, main = pp.Sequence(), pp.Sequence()
    first.read(paths[0])
    main.read(paths[1])

    assert [p.rsplit("/", 1)[-1] for p in paths] == ["scan.seq", "scan_main.seq"]
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
            "--n-dummy",
            "0",
        ],
    )

    assert status == 0
    assert path.read_text().startswith("# Pulseq sequence file")


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        ("gre2D_sequence", "--flip-angle-deg", "Excitation flip angle, in degrees."),
        ("fse3D_sequence", "--etl", "Echo train length: views per excitation."),
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


def test_the_binary_form_is_smaller_and_carries_no_signature(tmp_path):
    seq = gre()
    text, binary = tmp_path / "s.seq", tmp_path / "s.bin"

    signature = cli.write_sequence(seq, str(text))
    assert cli.write_sequence(seq, str(binary), offline=False) is None

    assert len(signature) == 32
    assert binary.stat().st_size < text.stat().st_size
    assert "[SIGNATURE]" in text.read_text()


def test_a_written_sequence_reads_back_as_itself(tmp_path):
    path = tmp_path / "gre.seq"
    cli.write_sequence(gre(), str(path))

    read_back = pp.Sequence()
    read_back.read(str(path))
    again = tmp_path / "again.seq"
    read_back.write(str(again))

    assert again.read_text() == path.read_text()


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
            "--n-dummy",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert path.exists()

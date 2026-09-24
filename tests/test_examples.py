"""Sequence examples, recorded prescriptions and generated command-line interfaces."""

import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: A prescription small enough to build in a moment, per example sequence.
SMALL = {
    "gre2D_sequence": {"n_x": 32, "n_y": 16, "n_slices": 1, "n_acs_y": 0},
    "gre3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8, "n_acs_y": 0, "n_acs_z": 0},
    "gre_multiecho2D_sequence": {"n_x": 32, "n_y": 16, "n_echoes": 3, "n_acs_y": 0},
    "gre_multiecho3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "n_echoes": 3,
        "n_acs_y": 0,
        "n_acs_z": 0,
    },
    "gre_stack_of_stars3D_sequence": {"n": 32, "n_z": 4},
    "gre_stack_of_spirals3D_sequence": {"n": 32, "n_z": 4, "n_shots": 4},
    "gre_stack_of_blades3D_sequence": {"n": 32, "n_z": 4, "blade_width": 8},
    "se_stack_of_stars3D_sequence": {"n": 32, "n_z": 4, "tr": None},
    "se_stack_of_spirals3D_sequence": {"n": 32, "n_z": 4, "n_shots": 4, "tr": None},
    "se_stack_of_blades3D_sequence": {"n": 32, "n_z": 4, "blade_width": 8, "tr": None},
    "zte3D_sequence": {"n": 32, "n_shots": 2, "n_dummy": 0},
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
    "epi2D_sequence": {"n_x": 32, "n_y": 16, "n_dummy": 0},
    "epi3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 4, "n_dummy": 0},
    "se2D_sequence": {"n_x": 32, "n_y": 16, "n_slices": 1, "n_acs_y": 0, "tr": None},
    "se3D_sequence": {
        "n_x": 32,
        "n_y": 8,
        "n_z": 4,
        "n_acs_y": 0,
        "n_acs_z": 0,
        "tr": None,
    },
    "mprage3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "ti": 100e-3,
        "tr": 300e-3,
        "n_dummy": 0,
    },
    "mprage_stack_of_spirals3D_sequence": {
        "n": 32,
        "n_z": 4,
        "n_shots": 4,
        "ti": 100e-3,
        "tr": 300e-3,
        "n_dummy": 0,
    },
    "mprage_stack_of_stars3D_sequence": {
        "n": 32,
        "n_z": 4,
        "ti": 100e-3,
        "tr": 500e-3,
        "n_dummy": 0,
    },
    "bssfp2D_sequence": {
        "n_x": 64,
        "n_y": 16,
        "readout_bandwidth_hz": 50e3,
        "n_dummy": 0,
    },
    "bssfp3D_sequence": {"n_x": 64, "n_y": 16, "n_z": 4},
    "fse3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "etl": 4,
        "te": None,
        "tr": 200e-3,
    },
}


def application(name):
    """The SequenceApp subclass a zoo entry's module defines."""
    module = getattr(sequences, name)
    (app,) = (
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, sequences.SequenceApp)
        and value.__module__ == module.__name__
    )
    return app


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
    """The 2D gradient echo, without dummies unless asked for."""
    app = sequences.gre2D_sequence.Gre2DApp
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
    app = gre_app(n_slices=8, tr=40e-3)

    assert len(app.packets) > 1
    for packet in app.packets:
        # Every slice of one packet is a whole packet count away from the next,
        # so no two neighbours in the slab are excited in the same packet.
        assert all(b - a >= len(app.packets) for a, b in pairwise(sorted(packet)))


def test_the_even_slices_of_a_packet_are_excited_before_the_odd_ones():
    app = gre_app(n_slices=10, tr=None)

    assert app.packets == [[0, 2, 4, 6, 8, 1, 3, 5, 7, 9]]


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

    _size, start = seq._detect_tr()

    assert start == 1


def test_every_slice_is_excited_at_the_repetition_time_asked_for():
    """Including the odd packet, which holds a slice more and waits less."""
    lines, tr = 8, 0.25
    app = gre_app(n_x=256, n_y=lines, n_slices=120, tr=tr, readout_oversampling=1.0)
    excited = np.asarray(app.design().rf_times()[0])

    assert len({len(packet) for packet in app.packets}) == 2  # the case worth asking

    at = 0
    for packet in app.packets:
        # A slice's repetition time is the gap between its own excitations.
        spacing = np.diff(excited[at : at + len(packet) * lines][:: len(packet)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(packet) * lines


def test_the_repetition_time_written_is_the_spacing_of_one_slices_excitations():
    app = gre_app(n_slices=3, tr=0.05)
    seq = app.design()
    excited = np.asarray(seq.rf_times()[0])

    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(0.05)
    assert excited[3] - excited[0] == pytest.approx(0.05)


def test_a_repetition_that_holds_whole_shots_takes_that_many_slices_a_packet():
    shot = gre_app().ro.duration + pp.Opts().block_duration_raster
    app = gre_app(n_slices=4, tr=2 * shot)

    assert [len(packet) for packet in app.packets] == [2, 2]


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
    app = gre_app(n_y=n_y, ry=ry, n_acs_y=0)

    assert n_y // 2 in app.lines
    assert all((line - n_y // 2) % ry == 0 for line in app.lines)


def test_partial_fourier_drops_lines_before_the_centre_only():
    app = gre_app(n_y=32, partial_fourier_y=0.75)

    assert app.lines == list(range(8, 32))


@pytest.mark.parametrize("name", ["partial_fourier_x", "partial_fourier_y"])
@pytest.mark.parametrize("fraction", [0.7, 1.01])
def test_a_partial_fourier_fraction_outside_its_range_is_refused(name, fraction):
    with pytest.raises(ValueError, match=name):
        gre_app(**{name: fraction})


def test_every_slice_is_rf_spoiled_by_its_own_excitation_count():
    """Consecutive excitations of one slice step their phase by a growing 117 degrees."""
    seq = gre_app(n_slices=3, n_y=8, tr=None).design()
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
    app = gre_app(n_y=32, ry=2, n_acs_y=8)

    assert list(app.lines[:8]) == sorted(app.calibration)


def test_a_fully_sampled_scan_has_no_calibration_block():
    app = gre_app(n_y=32, ry=1, n_acs_y=8)

    assert app.calibration == set()
    assert app.lines == list(range(32))


def test_each_acquisition_carries_the_line_and_slice_it_encodes():
    app = gre_app(n_dummy=2, n_y=16, n_slices=3, ry=2, n_acs_y=4)
    lin, slc, ima, seg = adc_labels(app.design(), "LIN", "SLC", "IMA", "SEG")

    expected = [
        (line, s) for packet in app.packets for line in app.lines for s in packet
    ]
    assert list(zip(lin, slc, strict=True)) == expected
    assert list(ima) == [int(line in app.calibration) for line, _ in expected]
    assert list(seg) == [1 - int(line in app.calibration) for line, _ in expected]


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


def test_a_parameter_carries_its_type_default_unit_and_description():
    parameters = sequences.gre2D_sequence.Gre2DApp.parameters()
    te, n_x = parameters["te"], parameters["n_x"]

    assert list(parameters) == list(sequences.gre2D_sequence.Gre2DApp.protocol())
    assert (te.type, te.default, te.optional, te.unit) == (float, 8e-3, True, "s")
    assert te.description.startswith("Echo time (s). ``None`` is as short as")
    assert (n_x.type, n_x.optional, n_x.unit) == (int, False, "")
    assert parameters["flip_angle_deg"].unit == "degrees"
    assert parameters["fov_x"].unit == parameters["fov_y"].unit == "m"


def test_the_choices_of_a_parameter_are_the_values_its_type_lists_in_order():
    parameters = sequences.fse3D_sequence.Fse3DApp.parameters()

    assert parameters["excitation"].choices == ("slab", "nonselective")
    assert parameters["wave"].choices == ("phase", "partition", "both")
    assert parameters["n_x"].choices == ()


@pytest.mark.parametrize("name", sequences.ZOO)
def test_every_prescribed_parameter_is_described_and_a_string_lists_its_choices(
    name,
):
    for parameter in application(name).parameters().values():
        assert parameter.description, parameter.name
        assert parameter.type is not None, parameter.name
        assert parameter.type is not str or parameter.choices, parameter.name


def test_the_units_the_shipped_sequences_state_are_the_package_units():
    stated = {
        parameter.unit
        for name in sequences.ZOO
        for parameter in application(name).parameters().values()
    }

    assert stated <= {"", "m", "s", "Hz", "degrees", "T/m", "beats per minute"}


class Pause(sequences.SequenceApp):
    """A pause of one TR per repetition, the TR its own shortest when not asked for."""

    MAX_GRAD = 40.0
    MAX_SLEW = 150.0
    SHORTEST = 10e-3

    def init_sequence(self, n: int = 3, tr: float | None = None, label: str = "a"):
        self.n, self.tr = n, self.SHORTEST if tr is None else tr
        if self.tr < self.SHORTEST:
            raise ValueError("the TR is shorter than a pause")
        self.resolve(tr=self.tr)
        self.loops = 0

    def loop(self):
        self.loops += 1
        for _ in range(self.n):
            self.kernel()

    def kernel(self):
        self.seq.add_block(pp.make_delay(self.tr))


def test_a_resolved_value_replaces_the_requested_one():
    assert Pause().resolved == {"n": 3, "tr": 10e-3, "label": "a"}
    assert Pause(n=2, tr=20e-3).resolved == {"n": 2, "tr": 20e-3, "label": "a"}


def test_resolving_a_name_init_sequence_does_not_take_is_refused():
    with pytest.raises(TypeError, match="has no parameter te"):
        Pause().resolve(te=1e-3)


def test_a_prescription_the_design_refuses_raises_on_construction():
    with pytest.raises(ValueError, match="shorter than a pause"):
        Pause(tr=5e-3)


def test_a_stated_scan_time_is_reported_without_designing():
    class Stated(Pause):
        def init_sequence(self, n: int = 3, tr: float | None = None):
            super().init_sequence(n, tr)
            self.duration = n * self.tr

    app = Stated(n=4)

    assert app.scan_time() == pytest.approx(40e-3)
    assert app.loops == 0


def test_without_a_stated_scan_time_the_whole_chain_is_designed_and_timed():
    class Prescanned(Pause):
        def prescans(self):
            return {"calibration": self.kernel}

    app = Prescanned(n=4)

    assert app.scan_time() == pytest.approx(50e-3)
    assert app.loops == 1
    assert app.seq.duration()[0] == pytest.approx(40e-3)


@pytest.mark.parametrize("name", sequences.ZOO)
def test_the_scan_time_is_the_time_the_designed_chain_plays(name):
    app = application(name)(pp.Opts(), **SMALL[name])
    stated = app.scan_time()
    designed = [app.design(prescan).duration()[0] for prescan in app.prescans()]

    assert stated == pytest.approx(sum(designed) + app.design().duration()[0])


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

"""A sequence written as a function: its labels, its files, its protocol and its command line."""

import inspect
from pathlib import Path

import pytest
from zoo import SMALL, parameters

import pypulseqpp as pp
from pypulseqpp import _ext, cli, sequences


def delay(seconds, name=None):
    """A sequence of one block that waits ``seconds``, recording ``name`` when given."""
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_delay(seconds))
    if name:
        seq.set_definition("Name", name)
    return seq


def events(written):
    """Each label event as its label, its type and its value."""
    return [(event.label, event.type, event.value) for event in written]


# -- labels ------------------------------------------------------------------

#: The labels a scan passes block by block: a first value, a repeated step, a
#: different step, an unchanged value, a change of ONCE, and a boolean.
SERIES = [
    {"LIN": 0, "SLC": 0, "ONCE": 1},
    {"LIN": 2, "SLC": 0, "ONCE": 1},
    {"LIN": 4, "SLC": 0},
    {"LIN": 6, "SLC": 1},
    {"LIN": 3, "SLC": 1},
    {"LIN": 3, "SLC": 1, "ONCE": 0},
    {"LIN": 4, "SLC": 1, "IMA": True},
]

SET, INC = "labelset", "labelinc"
WRITTEN = [
    [("LIN", SET, 0), ("SLC", SET, 0), ("ONCE", SET, 1)],
    [("LIN", SET, 2)],
    [("LIN", INC, 2)],
    [("LIN", INC, 2), ("SLC", SET, 1)],
    [("LIN", SET, 3)],
    [("LIN", SET, 3), ("SLC", SET, 1), ("ONCE", SET, 0)],
    [("LIN", SET, 4), ("IMA", SET, 1)],
]


def test_a_block_carries_only_the_label_events_it_changes():
    labels = sequences.Labels()

    assert [events(labels(**values)) for values in SERIES] == WRITTEN


def test_a_restart_writes_every_label_again_as_a_set():
    labels = sequences.Labels()
    for values in SERIES:
        labels(**values)
    assert events(labels(LIN=4)) == []

    labels.restart()

    assert events(labels(LIN=4)) == [("LIN", SET, 4)]


# -- writing -----------------------------------------------------------------


@pytest.mark.parametrize("offline", [True, False], ids=["text", "binary"])
def test_one_sequence_is_written_at_its_path(tmp_path, offline):
    path = tmp_path / "one.seq"

    written = sequences.write(path, delay(1e-3, "one"), offline=offline)

    assert written == [str(path)]
    assert list(tmp_path.iterdir()) == [path]
    assert _ext.is_binary(path.read_bytes()) is not offline
    assert pp.io.read(path).get_definition("NextSequence") == ""


@pytest.mark.parametrize("offline", [True, False], ids=["text", "binary"])
def test_a_chain_is_written_with_next_sequence_links(tmp_path, offline):
    """The first file is at the path; a later one is named by its ``Name`` or position."""
    chain = [delay(1e-3, "calibration"), delay(2e-3), delay(3e-3, "scan")]

    written = sequences.write(tmp_path / "scan.seq", chain, offline=offline)

    names = ["scan.seq", "scan_1.seq", "scan_scan.seq"]
    assert [Path(path).name for path in written] == names
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names)
    links = [pp.io.read(path).get_definition("NextSequence") for path in written]
    assert links == ["scan_1.seq", "scan_scan.seq", ""]
    assert [path.name for path, _ in pp.io.read_chain(written[0])] == names


@pytest.mark.parametrize(
    ("names", "files"),
    [
        (
            ["calibration", "scan", "calibration", "scan"],
            ["scan.seq", "scan_scan.seq", "scan_calibration.seq", "scan_scan_3.seq"],
        ),
        (
            ["calibration", "scan", "scan_3", "scan"],
            ["scan.seq", "scan_scan.seq", "scan_scan_3.seq", "scan_scan_3_3.seq"],
        ),
    ],
    ids=["alternating names", "a suffixed name that is taken"],
)
def test_a_chain_whose_names_repeat_writes_one_file_per_sequence(
    tmp_path, names, files
):
    """The first file stays at the path; a later file whose name is taken ends in its position."""
    durations = [1e-3, 2e-3, 3e-3, 4e-3]
    chain = [
        delay(seconds, name) for seconds, name in zip(durations, names, strict=True)
    ]

    written = sequences.write(tmp_path / "scan.seq", chain)

    assert [Path(path).name for path in written] == files
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(files)
    played = pp.io.read_chain(written[0])
    assert [path.name for path, _ in played] == files
    assert [seq.get_definition("Name") for _, seq in played] == names
    assert [seq.duration()[0] for _, seq in played] == pytest.approx(durations)


def test_a_link_is_recorded_in_the_files_and_not_on_the_sequences(tmp_path):
    chain = [delay(1e-3, "first"), delay(2e-3, "second")]

    sequences.write(tmp_path / "scan.seq", chain)

    assert all("NextSequence" not in seq.definitions for seq in chain)


def test_duration_sums_the_chain():
    one, two = delay(1e-3), delay(2e-3)

    assert sequences.duration(one) == pytest.approx(1e-3)
    assert sequences.duration([one, two]) == pytest.approx(3e-3)


# -- the command line --------------------------------------------------------


@pytest.fixture
def calls():
    """The arguments the function below was called with, one tuple per call."""
    return []


@pytest.fixture
def gradient_echo(calls):
    def gradient_echo(
        system, *, fov: float = 0.2, n_x: int = 64, spoil: bool = True
    ) -> pp.Sequence:
        """Gradient echo.

        Parameters
        ----------
        fov : float, default=0.2
            Field of view (m).
        n_x : int, default=64
            Readout samples.
        spoil : bool, default=True
            Spoil after the readout.
        """
        calls.append((system, fov, n_x, spoil))
        return delay(1e-3, "gradient_echo")

    return gradient_echo


def test_cli_flags_come_from_a_function_signature(
    gradient_echo, calls, capsys, tmp_path
):
    with pytest.raises(SystemExit):
        cli.run(gradient_echo, ["--help"])
    printed = " ".join(capsys.readouterr().out.split())

    assert "Gradient echo." in printed
    assert "--fov FOV Field of view (m)." in printed
    assert "--n-x N_X Readout samples." in printed
    assert "--no-spoil Spoil after the readout." in printed
    assert "--system" not in printed

    arguments = ["-o", str(tmp_path / "gre.seq"), "--n-x", "8", "--no-spoil"]
    status = cli.run(gradient_echo, [*arguments, "--max-grad-mtm", "30"])

    system, *protocol = calls.pop()
    limits = pp.Opts(max_grad=30, grad_unit="mT/m")
    assert (status, protocol) == (0, [0.2, 8, False])
    assert system.max_grad == limits.max_grad
    assert not calls


@pytest.mark.parametrize("flags", [[], ["--binary"]], ids=["text", "binary"])
def test_the_cli_writes_a_function_chain(tmp_path, capsys, flags):
    def chain(system, *, n: int = 1) -> list[pp.Sequence]:
        """Two sequences, the first a prescan.

        Parameters
        ----------
        n : int, default=1
            Delays in each sequence.
        """
        return [delay(1e-3 * n, "calibration"), delay(2e-3 * n, "scan")]

    path = tmp_path / "scan.seq"

    status = cli.run(chain, ["-o", str(path), "--n", "2", *flags])

    names = ["scan.seq", "scan_scan.seq"]
    written = [tmp_path / name for name in names]
    assert status == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)
    assert [path.name for path, _ in pp.io.read_chain(path)] == names
    assert all(_ext.is_binary(p.read_bytes()) is bool(flags) for p in written)
    assert capsys.readouterr().out.splitlines() == [
        f"Wrote sequence: {p}" for p in written
    ]


def test_the_cli_reports_every_sequence_a_function_returns(tmp_path, capsys):
    def chain(system) -> list[pp.Sequence]:
        """Two sequences."""
        return [delay(1e-3), delay(2e-3)]

    cli.run(chain, ["-o", str(tmp_path / "scan.seq"), "--report"])

    printed = capsys.readouterr().out
    assert printed.count("Sequence duration: 0.001000 s") == 1
    assert printed.count("Sequence duration: 0.002000 s") == 1


# -- a protocol read from a function -----------------------------------------


def test_the_system_is_not_a_protocol_parameter():
    def fn(system, fov: float = 0.2, *args, n: int | None = None, **extra):
        """Two parameters.

        Parameters
        ----------
        fov : float, default=0.2
            Field of view (m).
        n : int or None, default=None
            Matrix size.
        """

    protocol = sequences.parameters(fn)

    assert list(protocol) == ["fov", "n"]
    assert (protocol["fov"].type, protocol["fov"].unit) == (float, "m")
    assert (protocol["n"].type, protocol["n"].optional) == (int, True)


def test_a_parameter_carries_its_type_default_unit_and_description():
    main = sequences.gre2D_sequence.main
    protocol = sequences.parameters(main)
    te, n_x = protocol["te"], protocol["n_x"]

    assert list(protocol) == [
        name for name in inspect.signature(main).parameters if name != "system"
    ]
    assert (te.type, te.default, te.optional, te.unit) == (float, 8e-3, True, "s")
    assert te.description.startswith("Echo time (s). ``None`` is as short as")
    assert (n_x.type, n_x.optional, n_x.unit) == (int, False, "")
    assert protocol["flip_angle_deg"].unit == "degrees"
    assert protocol["fov_x"].unit == protocol["fov_y"].unit == "m"


@pytest.mark.parametrize(
    ("name", "choices"),
    [
        (
            "fse3D_sequence",
            {
                "excitation": ("slab", "nonselective"),
                "wave": ("phase", "partition", "both"),
                "n_x": (),
            },
        ),
        (
            "bssfp2D_sequence",
            {"gating": ("none", "retrospective", "prospective"), "n_x": ()},
        ),
    ],
)
def test_the_choices_of_a_parameter_are_the_values_its_type_lists_in_order(
    name, choices
):
    protocol = parameters(name)

    assert {key: protocol[key].choices for key in choices} == choices


@pytest.mark.parametrize("name", sequences.ZOO)
def test_every_prescribed_parameter_is_described_and_a_string_lists_its_choices(
    name,
):
    for parameter in parameters(name).values():
        assert parameter.description, parameter.name
        assert parameter.type is not None, parameter.name
        assert parameter.type is not str or parameter.choices, parameter.name


def test_the_units_the_shipped_sequences_state_are_the_package_units():
    stated = {
        parameter.unit
        for name in sequences.ZOO
        for parameter in parameters(name).values()
    }

    assert stated <= {"", "m", "s", "Hz", "degrees", "T/m", "beats per minute"}


@pytest.mark.parametrize("name", ["gre2D_sequence", "epi2D_sequence"])
def test_the_cli_writes_the_files_a_function_returns(tmp_path, name):
    """The files are those ``sequences.write`` writes for what the call returns."""
    main = getattr(sequences, name).main
    flags = [
        part
        for key, value in SMALL[name].items()
        for part in (f"--{key.replace('_', '-')}", str(value))
    ]
    for route in ("cli", "call"):
        (tmp_path / route).mkdir()

    status = cli.run(main, ["-o", str(tmp_path / "cli" / "scan.seq"), *flags])
    written = sequences.write(
        tmp_path / "call" / "scan.seq", main(pp.Opts(), **SMALL[name])
    )

    names = [Path(path).name for path in written]
    assert status == 0
    assert sorted(path.name for path in (tmp_path / "cli").iterdir()) == sorted(names)
    for file in names:
        assert (tmp_path / "cli" / file).read_bytes() == (
            tmp_path / "call" / file
        ).read_bytes()

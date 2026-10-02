"""A sequence written as a function writes the files its SequenceApp form writes.

``tests/legacy/`` holds the SequenceApp form of each such sequence, unedited.
For the default protocol, for each protocol in ``NON_DEFAULT`` and for system
limits above and below the sequence's own, the files of the application's chain
of prescans and main sequence are the files of the function's result, byte for
byte. The function takes the same system and protocol as the application and
documents them identically.
"""

import inspect
import re
from pathlib import Path

import pytest
from zoo import FUNCTIONS, LEGACY, SMALL, legacy_application

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp.sequences._app import _split_sections

#: The sequences whose SequenceApp form is kept in ``tests/legacy``.
LEGACY_NAMES = sorted(path.stem for path in LEGACY.glob("*_sequence.py"))

#: Protocols that exercise what the default one leaves out, per sequence. The
#: first of each changes every parameter; the others reach the packets, the
#: gating and the echo train a prescription of one slice or one echo does not.
NON_DEFAULT = {
    "gre2D_sequence": [
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "n_x": 32,
            "n_y": 24,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
        },
        # Seven slices dealt into packets of four and three.
        {
            "n_x": 32,
            "n_y": 16,
            "n_slices": 7,
            "slice_spacing": 1e-3,
            "te": 4e-3,
            "tr": 30e-3,
            "n_dummy": 3,
            "n_acs_y": 0,
        },
    ],
    "se2D_sequence": [
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "n_x": 32,
            "n_y": 24,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
        },
        # Five slices dealt into packets of two, two and one.
        {"n_x": 32, "n_y": 16, "n_slices": 5, "te": 22e-3, "tr": 70e-3, "n_acs_y": 0},
    ],
    "gre_multiecho2D_sequence": [
        # A bipolar train, which reads even echoes backwards and takes a full echo.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "n_x": 32,
            "n_y": 24,
            "n_slices": 2,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": 6e-3,
            "tr": None,
            "n_echoes": 3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "partial_fourier_y": 0.75,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
            "flyback": False,
        },
        # A monopolar train with a partial echo and a wait after every echo but
        # the last, in packets of three and two.
        {
            "n_x": 32,
            "n_y": 16,
            "n_slices": 5,
            "n_echoes": 3,
            "flyback": True,
            "echo_spacing": 6e-3,
            "te": 4e-3,
            "tr": 60e-3,
            "partial_fourier_x": 0.75,
            "n_dummy": 2,
            "n_acs_y": 0,
        },
    ],
    "bssfp2D_sequence": [
        # Prospective gating at a TR above the shortest.
        {
            "fov_x": 0.28,
            "fov_y": 0.25,
            "n_x": 64,
            "n_y": 16,
            "n_slices": 2,
            "slice_thickness": 5e-3,
            "slice_spacing": 1e-3,
            "flip_angle_deg": 60.0,
            "tr": 5e-3,
            "readout_bandwidth_hz": 50e3,
            "ry": 2,
            "partial_fourier_y": 0.75,
            "n_phases": 3,
            "n_dummy": 2,
            "readout_oversampling": 2.0,
            "n_acs_y": 4,
            "gating": "prospective",
            "heart_rate_bpm": 300.0,
            "views_per_segment": 4,
            "trigger_delay": 5e-3,
        },
        {
            "n_x": 64,
            "n_y": 16,
            "n_slices": 2,
            "readout_bandwidth_hz": 50e3,
            "n_dummy": 2,
            "n_acs_y": 0,
            "gating": "retrospective",
            "heart_rate_bpm": 120.0,
            "views_per_segment": 4,
        },
        # Ungated, three slices, undersampled with a calibration block.
        {
            "n_x": 64,
            "n_y": 24,
            "n_slices": 3,
            "slice_spacing": 2e-3,
            "tr": 4e-3,
            "readout_bandwidth_hz": 50e3,
            "ry": 2,
            "n_acs_y": 6,
            "partial_fourier_y": 0.75,
            "n_dummy": 0,
        },
    ],
    "gre_radial2D_sequence": [
        # A TE above the shortest, undersampled, in one packet.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": 5e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
        },
        # Seven slices dealt into packets of three, two and two.
        {"n": 32, "n_slices": 7, "te": None, "tr": 17e-3, "ry": 3, "n_dummy": 3},
        # A TR that holds one slice: three packets, no dummies.
        {"n": 16, "n_slices": 3, "te": None, "tr": 6e-3, "n_dummy": 0},
    ],
    "gre_spiral2D_sequence": [
        # A dual-density interleaf, every second one of four.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": 4e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "n_shots": 4,
            "density": "dual",
            "periphery_undersampling": 3.0,
            "transition_speed": 8.0,
        },
        # A variable-density interleaf, every third one of six, in packets of
        # three and two.
        {
            "n": 32,
            "n_slices": 5,
            "n_shots": 6,
            "ry": 3,
            "density": "variable",
            "periphery_undersampling": 2.5,
            "te": None,
            "tr": 20e-3,
            "n_dummy": 3,
        },
        # An odd number of constant-density interleaves, two slices, no dummies.
        {"n": 24, "n_slices": 2, "n_shots": 5, "te": None, "tr": None, "n_dummy": 0},
    ],
    "gre_propeller2D_sequence": [
        # Blades of eight lines, every second of the seven that cover the disc.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "blade_width": 8,
        },
        # Blades of seven lines, an odd width, in packets of three and two.
        {
            "n": 32,
            "n_slices": 5,
            "blade_width": 7,
            "te": 4e-3,
            "tr": 25e-3,
            "ry": 2,
            "n_dummy": 3,
        },
        # Blades as wide as the matrix, which two orientations cover.
        {
            "n": 24,
            "n_slices": 2,
            "blade_width": 24,
            "te": None,
            "tr": None,
            "n_dummy": 0,
        },
    ],
    "se_radial2D_sequence": [
        # A TE above the shortest, which delays the refocusing pulse.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": 20e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
        },
        # A TE off the block raster, which is rounded up, in packets of three
        # and two.
        {
            "n": 32,
            "n_slices": 5,
            "te": 16.253e-3,
            "tr": 60e-3,
            "ry": 3,
            "n_dummy": 1,
        },
        # A TR that holds one slice: three packets, no dummies.
        {"n": 24, "n_slices": 3, "te": None, "tr": 15e-3, "n_dummy": 0},
    ],
    "se_spiral2D_sequence": [
        # A dual-density interleaf, every second one of four, with a delayed
        # refocusing pulse.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": 20e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "n_shots": 4,
            "density": "dual",
            "periphery_undersampling": 3.0,
            "transition_speed": 8.0,
        },
        # A variable-density interleaf at a TE off the block raster, in packets
        # of three and two.
        {
            "n": 32,
            "n_slices": 5,
            "n_shots": 6,
            "ry": 3,
            "density": "variable",
            "periphery_undersampling": 2.5,
            "te": 16.253e-3,
            "tr": 70e-3,
            "n_dummy": 1,
        },
        # An odd number of constant-density interleaves, two slices, no dummies.
        {"n": 24, "n_slices": 2, "n_shots": 5, "te": None, "tr": None, "n_dummy": 0},
    ],
    "se_propeller2D_sequence": [
        # Blades of eight lines, every second of the seven that cover the disc,
        # at the shortest TE the slower readout admits.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "blade_width": 8,
        },
        # Blades of seven lines at a TE off the block raster, in packets of
        # three and two.
        {
            "n": 32,
            "n_slices": 5,
            "blade_width": 7,
            "te": 16.253e-3,
            "tr": 70e-3,
            "ry": 2,
            "n_dummy": 1,
        },
        # Blades as wide as the matrix, which two orientations cover.
        {
            "n": 24,
            "n_slices": 2,
            "blade_width": 24,
            "te": 20e-3,
            "tr": 100e-3,
            "n_dummy": 0,
        },
    ],
}

#: ``(max_grad in mT/m, max_slew in T/m/s)`` of a system above the limits a
#: sequence is designed under, and of one below them.
SYSTEMS = {"above": (150, 400), "below": (30, 100)}


def hardware(max_grad, max_slew):
    return pp.Opts(
        max_grad=max_grad, grad_unit="mT/m", max_slew=max_slew, slew_unit="T/m/s"
    )


def cases():
    """``(name, limits, protocol)``: the default, each non-default, each system."""
    found = []
    for name in LEGACY_NAMES:
        found.append(pytest.param(name, None, {}, id=f"{name}-default"))
        for index, protocol in enumerate(NON_DEFAULT.get(name, ())):
            found.append(
                pytest.param(name, None, protocol, id=f"{name}-non-default-{index}")
            )
        for label, limits in SYSTEMS.items():
            found.append(
                pytest.param(name, limits, SMALL[name], id=f"{name}-{label}-limits")
            )
    return found


def application_chain(name, limits, protocol):
    """The prescans and the main sequence of the SequenceApp form, in play order."""
    system = None if limits is None else hardware(*limits)
    app = legacy_application(name)(system, **protocol)
    return [app.design(prescan) for prescan in app.prescans()] + [app.design()]


def function_chain(name, limits, protocol):
    """What the function returns, as a list of sequences."""
    system = None if limits is None else hardware(*limits)
    result = getattr(sequences, name).main(system, **protocol)
    return [result] if isinstance(result, pp.Sequence) else list(result)


def written(directory, chain):
    """``(file name, bytes)`` of each file ``sequences.write`` writes for ``chain``."""
    directory.mkdir()
    paths = sequences.write(directory / "scan.seq", chain)
    return [(Path(path).name, Path(path).read_bytes()) for path in paths]


def normalised(text):
    """``text`` with a role on a class attribute as the literal of a module constant."""
    return re.sub(r":attr:`([^`]+)`", r"``\1``", text)


def test_every_sequence_with_a_legacy_application_is_written_as_a_function():
    assert LEGACY_NAMES
    assert set(LEGACY_NAMES) <= set(FUNCTIONS)


def test_every_sequence_with_a_legacy_application_has_a_non_default_protocol():
    assert set(NON_DEFAULT) == set(LEGACY_NAMES)
    assert all(NON_DEFAULT[name] for name in LEGACY_NAMES)


@pytest.mark.parametrize(("name", "limits", "protocol"), cases())
def test_a_function_writes_the_files_its_application_writes(
    tmp_path, name, limits, protocol
):
    from_application = written(
        tmp_path / "application", application_chain(name, limits, protocol)
    )
    from_function = written(
        tmp_path / "function", function_chain(name, limits, protocol)
    )

    assert [file for file, _ in from_function] == [file for file, _ in from_application]
    for (file, made), (_, wanted) in zip(from_function, from_application, strict=True):
        assert made == wanted, file


@pytest.mark.parametrize(("name", "limits", "protocol"), cases())
def test_every_protocol_compared_is_a_design_that_passes_its_timing_check(
    name, limits, protocol
):
    for seq in function_chain(name, limits, protocol):
        is_ok, errors = seq.check_timing()
        assert is_ok, errors


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_a_function_takes_the_system_and_the_protocol_its_application_takes(name):
    shipped = inspect.signature(getattr(sequences, name).main, eval_str=True)
    legacy = inspect.signature(legacy_application(name).function())

    def parameters(signature):
        return [
            (p.name, p.kind, p.default, p.annotation)
            for p in signature.parameters.values()
        ]

    assert parameters(shipped) == parameters(legacy)
    assert sequences.parameters(getattr(sequences, name).main) == sequences.parameters(
        legacy_application(name).function()
    )


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_a_function_documents_what_its_application_documents(name):
    """The protocol's own entries are compared through ``sequences.parameters``."""
    shipped = _split_sections(inspect.getdoc(getattr(sequences, name).main))
    legacy = _split_sections(inspect.getdoc(legacy_application(name).main))

    assert shipped[0] == normalised(legacy[0])
    assert shipped[1] == normalised(legacy[1])
    assert [heading for heading, _ in shipped[2]] == [
        heading for heading, _ in legacy[2]
    ]
    for (heading, body), (_, wanted) in zip(shipped[2], legacy[2], strict=True):
        if heading != "Parameters":
            assert body == normalised(wanted), heading

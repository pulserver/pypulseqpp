"""The shipped sequences acquire the same views, in the same order, with the same labels.

`sampling_regression.json` holds fingerprints of each configuration below:
the sampling attributes the application stores (lines, views, partitions,
calibration, trains, shots), every label evaluated at every acquisition, and
the results of direct calls of the support, traversal and EPI routines. The
sequence fingerprints and the lattice calls were first recorded from the
sampling routines before their rename and are unchanged by it. The Poisson
calls were recorded again after two support fixes: ``elliptical=False`` no
longer crops the draw to the inscribed ellipse, and an elliptical
calibration region no longer fills its bounding rectangle.

A deliberate change to one of these sequences is recorded again with

    python tests/test_sampling_regression.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

FINGERPRINTS = Path(__file__).with_name("sampling_regression.json")

#: The attributes a sequence application stores its sampling in.
ATTRIBUTES = (
    "lines",
    "views",
    "partitions",
    "calibration",
    "trains",
    "shots",
    "reference",
    "order",
)

CARTESIAN_2D = {"n_x": 32, "n_y": 16}
CARTESIAN_3D = {"n_x": 32, "n_y": 16, "n_z": 8}
ACCELERATED_3D = {
    **CARTESIAN_3D,
    "ry": 2,
    "rz": 2,
    "caipi_shift": 1,
    "n_acs_y": 4,
    "n_acs_z": 2,
    "partial_fourier_y": 0.75,
}
FSE = {"n_x": 32, "n_y": 16, "n_z": 8, "etl": 8, "te": 20e-3, "tr": 300e-3}
MPRAGE = {"n_x": 32, "n_y": 16, "n_z": 8, "ti": 100e-3, "tr": 300e-3}

#: ``name: (module, application class, keyword arguments)``.
CONFIGURATIONS = {
    "gre2D": ("gre2D_sequence", "Gre2DApp", CARTESIAN_2D),
    "gre2D-accelerated": (
        "gre2D_sequence",
        "Gre2DApp",
        {**CARTESIAN_2D, "ry": 2, "n_acs_y": 4, "partial_fourier_y": 0.75},
    ),
    "se2D-accelerated": (
        "se2D_sequence",
        "Se2DApp",
        {**CARTESIAN_2D, "ry": 2, "n_acs_y": 4, "tr": None},
    ),
    "bssfp2D-accelerated": (
        "bssfp2D_sequence",
        "Bssfp2DApp",
        {"n_x": 64, "n_y": 16, "readout_bandwidth_hz": 50e3, "ry": 2, "n_acs_y": 4},
    ),
    "gre_multiecho2D-accelerated": (
        "gre_multiecho2D_sequence",
        "GreMultiecho2DApp",
        {**CARTESIAN_2D, "n_echoes": 2, "ry": 2, "n_acs_y": 4},
    ),
    "gre3D": ("gre3D_sequence", "Gre3DApp", CARTESIAN_3D),
    "gre3D-caipi": ("gre3D_sequence", "Gre3DApp", ACCELERATED_3D),
    "gre3D-elliptical": (
        "gre3D_sequence",
        "Gre3DApp",
        {**ACCELERATED_3D, "elliptical_sampling": True, "elliptical_acs": True},
    ),
    "se3D-caipi": ("se3D_sequence", "Se3DApp", {**ACCELERATED_3D, "tr": None}),
    "bssfp3D-caipi": (
        "bssfp3D_sequence",
        "Bssfp3DApp",
        {"n_x": 64, "n_y": 16, "n_z": 4, "ry": 2, "rz": 2, "caipi_shift": 1},
    ),
    "gre_multiecho3D-caipi": (
        "gre_multiecho3D_sequence",
        "GreMultiecho3DApp",
        {**ACCELERATED_3D, "n_echoes": 2},
    ),
    "mprage3D-radial": ("mprage3D_sequence", "Mprage3DApp", MPRAGE),
    "mprage3D-caipi": (
        "mprage3D_sequence",
        "Mprage3DApp",
        {**MPRAGE, "ry": 2, "rz": 2, "caipi_shift": 1, "n_acs_y": 4, "n_acs_z": 2},
    ),
    "mprage3D-shuffling": (
        "mprage3D_sequence",
        "Mprage3DApp",
        {
            **MPRAGE,
            "ry": 2,
            "rz": 2,
            "n_acs_y": 4,
            "n_acs_z": 2,
            "ordering": "shuffling",
        },
    ),
    "fse3D-radial": (
        "fse3D_sequence",
        "Fse3DApp",
        {**FSE, "ry": 2, "rz": 2, "n_acs_y": 4, "n_acs_z": 2},
    ),
    "fse3D-shuffling": (
        "fse3D_sequence",
        "Fse3DApp",
        {**FSE, "ry": 2, "rz": 2, "n_acs_y": 4, "n_acs_z": 2, "ordering": "shuffling"},
    ),
    "fse3D-individual": (
        "fse3D_sequence",
        "Fse3DApp",
        {**FSE, "tr_periphery": 150e-3, "etl_periphery": 5},
    ),
    "gre_stack_of_stars3D": (
        "gre_stack_of_stars3D_sequence",
        "GreStackOfStars3DApp",
        {"n": 32, "n_z": 8, "rz": 2, "n_acs_z": 2},
    ),
    "se_stack_of_stars3D": (
        "se_stack_of_stars3D_sequence",
        "SeStackOfStars3DApp",
        {"n": 32, "n_z": 8, "rz": 2, "n_acs_z": 2, "tr": None},
    ),
    "gre_stack_of_spirals3D": (
        "gre_stack_of_spirals3D_sequence",
        "GreStackOfSpirals3DApp",
        {"n": 32, "n_z": 8, "n_shots": 4, "rz": 2, "n_acs_z": 2},
    ),
    "se_stack_of_spirals3D": (
        "se_stack_of_spirals3D_sequence",
        "SeStackOfSpirals3DApp",
        {"n": 32, "n_z": 8, "n_shots": 4, "rz": 2, "n_acs_z": 2, "tr": None},
    ),
    "gre_stack_of_blades3D": (
        "gre_stack_of_blades3D_sequence",
        "GreStackOfBlades3DApp",
        {"n": 32, "n_z": 8, "blade_width": 8, "rz": 2, "n_acs_z": 2},
    ),
    "se_stack_of_blades3D": (
        "se_stack_of_blades3D_sequence",
        "SeStackOfBlades3DApp",
        {"n": 32, "n_z": 8, "blade_width": 8, "rz": 2, "n_acs_z": 2, "tr": None},
    ),
    "mprage_stack_of_stars3D": (
        "mprage_stack_of_stars3D_sequence",
        "MprageStackOfStars3DApp",
        {"n": 32, "n_z": 8, "rz": 2, "n_acs_z": 2, "ti": 100e-3, "tr": 500e-3},
    ),
    "mprage_stack_of_spirals3D": (
        "mprage_stack_of_spirals3D_sequence",
        "MprageStackOfSpirals3DApp",
        {
            "n": 32,
            "n_z": 8,
            "n_shots": 4,
            "rz": 2,
            "n_acs_z": 2,
            "ti": 100e-3,
            "tr": 300e-3,
        },
    ),
    "se_epi_propeller2D": (
        "se_epi_propeller2D_sequence",
        "SeEpiPropeller2DApp",
        {
            "n_x": 32,
            "blade_width": 8,
            "n_blades": 4,
            "n_slices": 3,
            "slice_order": "interleaved",
            "te": None,
            "tr": None,
        },
    ),
    "epi2D-segmented": (
        "epi2D_sequence",
        "Epi2DApp",
        {"n_x": 32, "n_y": 16, "n_dummy": 0, "n_shots": 2, "ry": 2},
    ),
    "epi3D-caipi": (
        "epi3D_sequence",
        "Epi3DApp",
        {"n_x": 32, "n_y": 16, "n_z": 8, "n_dummy": 0, "ry": 2, "rz": 2},
    ),
}


def _digest(value) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def _plain(value):
    """Nested lists, tuples and arrays of integers as plain Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def fingerprint(name: str) -> dict:
    """Digest the stored sampling attributes and the acquisition labels."""
    import pypulseqpp as pp
    from pypulseqpp import sequences

    module, application, kwargs = CONFIGURATIONS[name]
    app = getattr(getattr(sequences, module), application)(pp.Opts(), **kwargs)
    seq = app.design()
    found = {
        attribute: _digest(_plain(getattr(app, attribute)))
        for attribute in ATTRIBUTES
        if not callable(getattr(app, attribute, print))
    }
    labels = seq.evaluate_labels(evolution="adc")
    for label in sorted(labels):
        found[f"label:{label}"] = _digest(np.atleast_1d(labels[label]).tolist())
    return found


#: Direct calls of the support, traversal and EPI routines: ``name: (routine, kwargs)``.
CALLS = {
    **{
        f"axis-{n}-{r}-{acs}-{pf}": (
            "make_cartesian_axis_sampling",
            {"n": n, "acceleration": r, "n_acs": acs, "partial_fourier": pf},
        )
        for n in (15, 16)
        for r in (1, 2, 3)
        for acs in (0, 5)
        for pf in (1.0, 0.7)
    },
    **{
        f"plane-{r}-{shift}-{pf}-{ellipse}-{scheme}": (
            "make_cartesian_plane_sampling",
            {
                "shape": (24, 20),
                "acceleration": r,
                "n_acs": (6, 4),
                "caipi_shift": shift,
                "partial_fourier": pf,
                "elliptical": ellipse,
                "elliptical_acs": ellipse,
                "sampling": scheme,
                "seed": 3,
            },
        )
        for r in ((1, 1), (2, 2), (3, 2))
        for shift in (0, 1)
        for pf in ((1.0, 1.0), (0.75, 0.8))
        for ellipse in (False, True)
        for scheme in ("lattice", "poisson")
    },
    **{
        f"traversal-{order}-{n}": (
            "make_traversal_order",
            {"n": n, "order": order, "seed": 2},
        )
        for n in (0, 1, 6, 7)
        for order in (
            "sequential",
            "reverse",
            "interleaved",
            "center_out",
            "outside_in",
            "random",
        )
    },
    **{
        f"epi-{etl}-{scheme}-{r}-{segments}": (
            "make_epi_shot_offsets",
            {
                "etl": etl,
                "scheme": scheme,
                "acceleration": r,
                "segments": segments,
                "partition_acceleration": 3,
                "caipi_shift": 1,
                "extent": 12 if scheme == "zigzag" else None,
            },
        )
        for etl in (1, 9)
        for scheme in ("linear", "caipi", "zigzag")
        for r in (1, 2)
        for segments in (1, 3)
    },
}


def call(name: str):
    """Digest one routine's result."""
    import pypulseqpp as pp

    routine, kwargs = CALLS[name]
    return _digest(_plain(getattr(pp, routine)(**kwargs)))


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_sampling_routine_returns_its_recorded_result(name):
    expected = json.loads(FINGERPRINTS.read_text())["calls"][name]

    assert call(name) == expected


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_a_shipped_sequence_samples_and_labels_as_recorded(name):
    expected = json.loads(FINGERPRINTS.read_text())[name]

    assert fingerprint(name) == expected


if __name__ == "__main__":
    recorded = {name: fingerprint(name) for name in sorted(CONFIGURATIONS)}
    recorded["calls"] = {name: call(name) for name in sorted(CALLS)}
    FINGERPRINTS.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
    sys.exit(0)

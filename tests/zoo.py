"""Small prescriptions of the shipped sequences, and what each module defines.

A shipped sequence is written either as a ``SequenceApp`` subclass or as a
function. ``APPLICATIONS`` and ``FUNCTIONS`` partition ``sequences.ZOO`` by
which. The ``SequenceApp`` form of a sequence written as a function is kept in
``tests/legacy/`` and reached through ``legacy_application``.
"""

import functools
import importlib.util
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

from pypulseqpp import sequences

#: Where the ``SequenceApp`` form of each sequence written as a function is kept.
LEGACY = Path(__file__).parent / "legacy"

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


def _defined_in(module):
    """The ``SequenceApp`` subclasses a module defines itself."""
    return [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, sequences.SequenceApp)
        and value.__module__ == module.__name__
    ]


def is_application(name):
    """Whether the sequence ``name`` is a ``SequenceApp`` subclass, not a function."""
    return bool(_defined_in(getattr(sequences, name)))


def application(name):
    """The ``SequenceApp`` subclass the module ``name`` defines.

    Refused for a sequence written as a function, whose ``SequenceApp`` form is
    ``legacy_application``.
    """
    found = _defined_in(getattr(sequences, name))
    if not found:
        raise LookupError(f"{name} is a sequence function, not a SequenceApp subclass")
    return found[0]


@functools.cache
def legacy_application(name):
    """The ``SequenceApp`` subclass ``tests/legacy`` keeps for the sequence ``name``."""
    spec = importlib.util.spec_from_file_location(
        f"legacy_{name}", LEGACY / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    (app,) = _defined_in(module)
    return app


def function(name):
    """The sequence function of the sequence ``name``, ``function(system, **protocol)``."""
    if is_application(name):
        return application(name).function()
    return getattr(sequences, name).main


def parameters(name):
    """The protocol of the sequence ``name``, as ``sequences.parameters`` reads it."""
    return sequences.parameters(function(name))


def packets(seq):
    """The slices each packet of a multi-slice sequence excites, in play order.

    Read from the ``LIN`` and ``SLC`` labels of the acquisitions. Every line of a
    packet is acquired for every slice of it before the next line, so the slices
    that share a ``LIN`` are the packet's, and a change of them is the next
    packet. The echoes of one excitation share its slice.
    """
    labels = seq.evaluate_labels(evolution="adc")
    lin, slc = (np.atleast_1d(labels[name]) for name in ("LIN", "SLC"))
    runs = [
        [s for s, _ in groupby(s for _, s in run)]
        for _, run in groupby(zip(lin, slc, strict=True), key=lambda pair: pair[0])
    ]
    return [run for i, run in enumerate(runs) if not i or run != runs[i - 1]]


#: The shipped sequences written as ``SequenceApp`` subclasses.
APPLICATIONS = tuple(name for name in sequences.ZOO if is_application(name))

#: The shipped sequences written as functions.
FUNCTIONS = tuple(name for name in sequences.ZOO if not is_application(name))

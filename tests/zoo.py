"""Small prescriptions of the shipped sequences, and helpers the sequence tests share."""

from itertools import groupby

import numpy as np

from pypulseqpp import sequences

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
        "readout_bandwidth_hz": 50e3,
    },
    "epi2D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "readout_bandwidth_hz": 50e3,
        "n_dummy": 0,
    },
    "epi3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 4,
        "readout_bandwidth_hz": 50e3,
        "n_dummy": 0,
    },
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


def parameters(name):
    """The protocol of the sequence ``name``, as ``sequences.parameters`` reads it."""
    return sequences.parameters(getattr(sequences, name).main)


def chain(result):
    """The sequences a sequence function returned, in play order, main sequence last."""
    return result if isinstance(result, list) else [result]


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

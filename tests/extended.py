"""Sequences using the 1.5.1 event kinds, built on the core directly.

Rotations, RF shims and labels outside Pulseq's own table arrived in 1.5.1
and 1.5.2, and the reference toolbox this package is tested against does not
implement them -- it has no `make_rotation`, no `make_rf_shim` and no way to
add a label name. So these sequences are built against the core itself, and
what holds them is a round trip rather than a comparison with another
writer's bytes.

They are deliberately small: what is being tested is that each section is
written, read and registered again, not that the numbers in it mean anything.
"""

from __future__ import annotations

import numpy as np

from pypulseqpp import _ext

#: A label Pulseq does not define, which is what makes a file revision 1.5.2.
CUSTOM_LABEL = "SPARKLE"


def _readout(sequence):
    """One readout worth of events, so a block has something to play."""
    gradient = sequence.register_trap(np.array([2000.0, 1e-4, 2e-3, 1e-4, 0.0]))
    adc = np.zeros(8)
    adc[0], adc[1] = 128.0, 1e-5
    return gradient, sequence.register_adc(adc)


def _chain(sequence, name, reference, next_link=0):
    return sequence.chain_extension(
        sequence.extension_type_id(name), reference, next_link
    )


def rotations():
    """A radial shot turned by a quaternion per view."""
    sequence = _ext.Sequence()
    gradient, adc = _readout(sequence)
    for view in range(8):
        angle = view * np.pi / 8
        rotation = sequence.register_rotation(
            np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        )
        sequence.add_block(
            0, gradient, 0, 0, adc, _chain(sequence, "ROTATIONS", rotation), 2.2e-3
        )
    return sequence


def rf_shims():
    """A parallel-transmit pulse, shimmed differently on each of four channels."""
    sequence = _ext.Sequence()
    magnitude = sequence.register_shape(64, np.linspace(0, 1, 64))
    phase = sequence.register_shape(64, np.zeros(64))
    time = sequence.register_shape(64, np.arange(64) * 1e-5)
    row = np.zeros(10)
    row[0], row[1], row[2], row[3], row[5] = 500.0, magnitude, phase, time, 1e-4
    pulse = sequence.register_rf(row, "e")

    for shot in range(3):
        weights = np.empty(8)
        weights[0::2] = 1.0 - 0.1 * shot  # a magnitude per channel
        weights[1::2] = np.arange(4) * 0.25 * shot  # and a phase
        shim = sequence.register_rf_shim(weights)
        sequence.add_block(pulse, 0, 0, 0, 0, _chain(sequence, "RF_SHIMS", shim), 1e-3)
    return sequence


def custom_labels():
    """A counter Pulseq does not define, beside one it does."""
    sequence = _ext.Sequence()
    gradient, adc = _readout(sequence)
    for shot in range(4):
        line = sequence.register_label_set(shot, sequence.label_id("LIN"))
        sparkle = sequence.register_label_set(shot * 2, sequence.label_id(CUSTOM_LABEL))
        chain = _chain(
            sequence, "LABELSET", sparkle, _chain(sequence, "LABELSET", line)
        )
        sequence.add_block(0, gradient, 0, 0, adc, chain, 2.2e-3)
    return sequence


def every_extension():
    """One sequence carrying all four extension kinds at once."""
    sequence = _ext.Sequence()
    gradient, adc = _readout(sequence)
    magnitude = sequence.register_shape(32, np.linspace(0, 1, 32))
    phase = sequence.register_shape(32, np.zeros(32))
    time = sequence.register_shape(32, np.arange(32) * 1e-5)
    row = np.zeros(10)
    row[0], row[1], row[2], row[3], row[5] = 400.0, magnitude, phase, time, 1e-4
    pulse = sequence.register_rf(row, "e")

    trigger = sequence.register_trigger(np.array([2.0, 1.0, 0.0, 1e-3]))
    delay = _ext.SoftDelay(num=1, offset=-2e-4, factor=1.0, hint="TE")

    for shot in range(3):
        angle = shot * np.pi / 3
        rotation = sequence.register_rotation(
            np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        )
        shim = sequence.register_rf_shim(np.array([1.0, 0.0, 0.9, 0.5 * shot]))
        label = sequence.register_label_set(shot, sequence.label_id(CUSTOM_LABEL))

        sequence.add_block(pulse, 0, 0, 0, 0, _chain(sequence, "RF_SHIMS", shim), 1e-3)
        sequence.add_block(0, 0, 0, 0, 0, _chain(sequence, "TRIGGERS", trigger), 1e-3)
        sequence.add_block(
            0,
            0,
            0,
            0,
            0,
            _chain(sequence, "DELAYS", sequence.register_soft_delay(delay)),
            5e-4,
        )
        sequence.add_block(
            0,
            gradient,
            0,
            0,
            adc,
            _chain(
                sequence, "ROTATIONS", rotation, _chain(sequence, "LABELSET", label)
            ),
            2.2e-3,
        )
    return sequence


#: Every extended sequence, by name.
ZOO = {
    "rotations": rotations,
    "rf_shims": rf_shims,
    "custom_labels": custom_labels,
    "every_extension": every_extension,
}

"""The RF events a sequence plays, grouped into the pulses they play."""

from __future__ import annotations

__all__ = ["rf_instances"]

from types import SimpleNamespace

import numpy as np

from ._ptx import split_ptx_pulse
from ._results import RfDefinition, RfInstances


def rf_instances(seq) -> RfInstances:
    """Take a sequence's RF instances; see :meth:`pypulseqpp.Sequence.rf_instances`."""
    tables = seq.libraries()
    ids = tables.blocks[:, 0]
    played = np.flatnonzero(ids)
    amplitude_hz = tables.rf[ids[played] - 1, 0]

    # Which instances play one pulse is read off a collapsed copy, in which
    # equal pulses registered separately, and equal shims, are one row.
    collapsed = seq._native.copy()
    collapsed.remove_duplicates()
    pulse = np.asarray(collapsed.rf_definitions(), dtype=np.int64)[
        collapsed.block_events()[played, 0] - 1
    ]
    shim = collapsed.block_shims()[played]
    key = pulse * (collapsed.num_rf_shims() + 1) + shim
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    by_first_play = np.argsort(first)
    definition = np.argsort(by_first_play)[inverse]

    # The base instance of a definition is its first with a nonzero amplitude.
    base = first[by_first_play]
    nonzero = np.flatnonzero(amplitude_hz)
    numbers, at = np.unique(definition[nonzero], return_index=True)
    base[numbers] = nonzero[at]
    base_hz = amplitude_hz[base]
    scale = base_hz[definition]
    amplitude = np.divide(
        amplitude_hz, scale, out=np.ones_like(amplitude_hz), where=scale != 0
    )

    flips = seq.rf_flip_angles()
    definitions = tuple(
        _definition(seq, tables, flips, int(played[instance]) + 1, float(hz))
        for instance, hz in zip(base, base_hz, strict=True)
    )
    return RfInstances(definitions, definition, amplitude)


def _definition(seq, tables, flips, block, amplitude_hz) -> RfDefinition:
    """Take the pulse a block plays, at unit amplitude when ``amplitude_hz`` is zero."""
    event = seq.get_block(block)
    rf = event.rf
    times = np.asarray(rf.t)
    channels = split_ptx_pulse(
        SimpleNamespace(t=times, signal=rf.magnitude * np.exp(2j * np.pi * rf.phase))
    )
    if event.rf_shim is not None:
        weights = event.rf_shim.shim_vector
        if channels.shape[0] not in (1, weights.size):
            raise ValueError(
                f"block {block} plays {channels.shape[0]} transmit channels "
                f"and its RF shim weighs {weights.size}"
            )
        channels = channels * weights[:, None]
    rf_hz = (amplitude_hz or 1.0) * channels
    peak = np.abs(rf_hz).max()
    rf_id = tables.blocks[block - 1, 0]
    return RfDefinition(
        waveform=rf_hz / peak if peak else rf_hz,
        peak_hz=float(peak) if amplitude_hz else 0.0,
        time=times[: channels.shape[1]],
        delay=float(rf.delay),
        center=float(rf.center),
        use=tables.rf_use[rf_id - 1],
        flip_deg=float(flips[rf_id - 1]),
    )

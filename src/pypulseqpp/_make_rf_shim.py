"""RF shim extension event constructor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

__all__ = ["make_rf_shim"]


def make_rf_shim(shim_vector) -> SimpleNamespace:
    """Create complex per-transmit-channel weights for a block's RF envelope.

    Parameters
    ----------
    shim_vector : array_like
        Complex per-channel weights, one per transmit channel.

    Returns
    -------
    types.SimpleNamespace
        RF shim extension event (``type == 'rf_shim'``) with the weights
        stored as ``complex128``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> event = pp.make_rf_shim([1.0, 1j, -1.0, -1j])
    >>> event.type, event.shim_vector.dtype
    ('rf_shim', dtype('complex128'))

    Attach it alongside the RF event of a block::

        seq.add_block(pulse.rf, pp.make_rf_shim(weights))
    """
    event = SimpleNamespace()
    event.type = "rf_shim"
    event.shim_vector = np.asarray(shim_vector, dtype=np.complex128)
    return event

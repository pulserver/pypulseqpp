"""RF shim extension event constructor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

__all__ = ["make_rf_shim"]


def make_rf_shim(shim_vector) -> SimpleNamespace:
    """Construct an RF shim extension event.

    The weights multiply the block's RF waveform per transmit channel.

    Parameters
    ----------
    shim_vector : array_like
        Complex weight of each transmit channel (dimensionless).

    Returns
    -------
    types.SimpleNamespace
        Event with ``type == 'rf_shim'`` and ``shim_vector`` as
        ``complex128``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> event = pp.make_rf_shim([1.0, 1j, -1.0, -1j])
    >>> event.type, event.shim_vector.dtype
    ('rf_shim', dtype('complex128'))

    Played in the block of the RF event it weights::

        seq.add_block(pulse.rf, pp.make_rf_shim(weights))
    """
    event = SimpleNamespace()
    event.type = "rf_shim"
    event.shim_vector = np.asarray(shim_vector, dtype=np.complex128)
    return event

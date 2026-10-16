"""How many readouts a receive-gain prescan acquires to play whole repetitions."""

from __future__ import annotations

__all__ = ["gain_calibration_readouts"]

import numpy as np

import pypulseqpp as pp


def gain_calibration_readouts(seq: pp.Sequence, repetitions: int = 1) -> int:
    """Return the readouts from the first block to the end of the ``repetitions``-th imaging repetition.

    A receive-gain prescan plays the sequence from its first block and stops
    after the ``NumGainCalibrationReadouts`` definition's count of readouts,
    so a count it plays through whole repetitions of
    :meth:`~pypulseqpp.Sequence.repetition`, found on the sequence as it is
    written, its duplicate events merged, sets the gain on the signal the
    scan acquires. An imaging repetition is one that acquires a readout not
    flagged ``NAV``; the dummy repetitions before it play and their readouts,
    navigators included, are counted with it.

    Parameters
    ----------
    seq : pypulseqpp.Sequence
        The designed sequence, its definitions set.
    repetitions : int, default=1
        Imaging repetitions the prescan plays, at least one.

    Returns
    -------
    int
        Readouts the prescan acquires.

    Raises
    ------
    ValueError
        If ``repetitions`` is below one, or the sequence has fewer imaging
        repetitions.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre2D_sequence(n_y=8, n_dummy=2)
    >>> sequences.gain_calibration_readouts(seq, repetitions=3)
    3
    """
    if repetitions < 1:
        raise ValueError(f"a prescan plays at least one repetition, not {repetitions}")
    # The repetition of the file the sequence is written as, its events merged.
    seq = seq.remove_duplicates()
    size, _ = seq.repetition()
    acquires = np.asarray(seq._native.block_events())[:, 4] > 0
    readouts = np.flatnonzero(acquires)
    navigator = seq.evaluate_labels(evolution="adc").get("NAV", 0)
    navigator = np.broadcast_to(np.asarray(navigator) != 0, readouts.shape)
    imaging = np.unique(readouts[~navigator] // size) if size else np.array([], int)
    if imaging.size < repetitions:
        raise ValueError(
            f"the sequence has {imaging.size} imaging repetitions of {size} "
            f"blocks, fewer than the {repetitions} a prescan plays"
        )
    end = (int(imaging[repetitions - 1]) + 1) * size
    return int(np.count_nonzero(acquires[:end]))

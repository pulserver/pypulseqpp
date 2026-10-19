"""How many readouts a receive-gain prescan acquires to play whole repetitions."""

from __future__ import annotations

__all__ = ["GAIN_CALIBRATION_DURATION", "gain_calibration_readouts"]

import numpy as np

import pypulseqpp as pp

#: Shortest receive-gain calibration (s): the imaging repetitions a prescan
#: plays last at least this long, so a short repetition is played many times
#: and a long one once.
GAIN_CALIBRATION_DURATION = 0.5


def gain_calibration_readouts(
    seq: pp.Sequence,
    repetitions: int | None = None,
    duration: float = GAIN_CALIBRATION_DURATION,
) -> int:
    """Return the readouts from the first block to the end of the prescan's last imaging repetition.

    A receive-gain prescan plays the sequence from its first block and stops
    after the ``NumGainCalibrationReadouts`` definition's count of readouts,
    so a count it plays through whole repetitions of
    :meth:`~pypulseqpp.Sequence.repetition`, found on the sequence as it is
    written, its duplicate events merged, sets the gain on the signal the
    scan acquires. An imaging repetition is one that acquires a readout not
    flagged ``NAV``; the dummy repetitions before it play and their readouts,
    navigators included, are counted with it. The prescan plays the fewest
    imaging repetitions that last ``duration`` together, or all of a shorter
    scan's, unless ``repetitions`` names how many.

    Parameters
    ----------
    seq : pypulseqpp.Sequence
        The designed sequence, its definitions set.
    repetitions : int, default=None
        Imaging repetitions the prescan plays, at least one; ``None`` takes
        as many as ``duration`` asks.
    duration : float, default=GAIN_CALIBRATION_DURATION
        Shortest time (s) the imaging repetitions of the prescan last.

    Returns
    -------
    int
        Readouts the prescan acquires.

    Raises
    ------
    ValueError
        If ``repetitions`` is below one, or the sequence has fewer imaging
        repetitions than it names.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre2D_sequence(n_y=8, n_dummy=2)
    >>> sequences.gain_calibration_readouts(seq, repetitions=3)
    3
    """
    if repetitions is not None and repetitions < 1:
        raise ValueError(f"a prescan plays at least one repetition, not {repetitions}")
    # The repetition of the file the sequence is written as, its events merged.
    seq = seq.remove_duplicates()
    size, _ = seq.repetition()
    acquires = np.asarray(seq._native.block_events())[:, 4] > 0
    readouts = np.flatnonzero(acquires)
    navigator = seq.evaluate_labels(evolution="adc").get("NAV", 0)
    navigator = np.broadcast_to(np.asarray(navigator) != 0, readouts.shape)
    imaging = np.unique(readouts[~navigator] // size) if size else np.array([], int)
    if repetitions is None:
        # The fewest imaging repetitions that last `duration` together.
        durations = np.asarray(seq._native.block_durations())
        lasting = np.cumsum(
            [durations[r * size : (r + 1) * size].sum() for r in imaging]
        )
        repetitions = max(
            1,
            min(int(np.searchsorted(lasting, duration * (1 - 1e-9))) + 1, imaging.size),
        )
    if imaging.size < repetitions:
        raise ValueError(
            f"the sequence has {imaging.size} imaging repetitions of {size} "
            f"blocks, fewer than the {repetitions} a prescan plays"
        )
    end = (int(imaging[repetitions - 1]) + 1) * size
    return int(np.count_nonzero(acquires[:end]))

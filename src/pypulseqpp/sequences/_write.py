"""Writing and timing the sequence, or the chain of sequences, a sequence function returns."""

from __future__ import annotations

from pathlib import Path

import pypulseqpp as pp

__all__ = ["duration", "write"]


def _chain(result: pp.Sequence | list[pp.Sequence]) -> list[pp.Sequence]:
    return [result] if isinstance(result, pp.Sequence) else list(result)


def write(
    path: str | Path,
    result: pp.Sequence | list[pp.Sequence],
    *,
    offline: bool = True,
) -> list[str]:
    """Write a sequence, or a chain of sequences, as Pulseq files.

    One sequence is written at ``path``. A chain is a list of sequences played
    in order, prescans first and the main sequence last: the first is written
    at ``path`` and each later one beside it as ``<stem>_<Name>.seq``, where
    ``<Name>`` is the sequence's ``Name`` definition, or its position in the
    chain, counted from 0 at the first, when it has none. Each file but the
    last records the next file's name as its ``NextSequence`` definition, so
    an interpreter plays the chain as one scan while every file stays one
    repeating unit. The definition is recorded in the file and not on the
    sequence.

    Parameters
    ----------
    path : str or pathlib.Path
        Where the first file of the chain is written.
    result : pypulseqpp.Sequence or list of pypulseqpp.Sequence
        What a sequence function returns.
    offline : bool, default=True
        Write signed text. False writes the binary form;
        :func:`pypulseqpp.io.write` takes this the other way round, as
        ``binary``.

    Returns
    -------
    list of str
        The written paths, in play order.

    See Also
    --------
    pypulseqpp.io.read_chain : Read the files of a chain in play order.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> import pypulseqpp as pp
    >>> from pypulseqpp import sequences
    >>> def one_delay(name):
    ...     seq = pp.Sequence(pp.Opts())
    ...     seq.add_block(pp.make_delay(1e-3))
    ...     seq.set_definition("Name", name)
    ...     return seq
    >>> directory = pathlib.Path(tempfile.mkdtemp())
    >>> written = sequences.write(
    ...     directory / "scan.seq", [one_delay("calibration"), one_delay("scan")]
    ... )
    >>> [pathlib.Path(path).name for path in written]
    ['scan.seq', 'scan_scan.seq']
    >>> [str(path.name) for path, _ in pp.io.read_chain(written[0])]
    ['scan.seq', 'scan_scan.seq']
    """
    chain = _chain(result)
    path = Path(path)
    paths = [
        path
        if index == 0
        else path.with_name(f"{path.stem}_{seq.get_definition('Name') or index}.seq")
        for index, seq in enumerate(chain)
    ]
    for index, (seq, target) in enumerate(zip(chain, paths, strict=True)):
        written = seq.remove_duplicates()
        if index + 1 < len(chain):
            written.set_definition("NextSequence", paths[index + 1].name)
        pp.io.write(written, target, binary=not offline, remove_duplicates=False)
    return [str(target) for target in paths]


def duration(result: pp.Sequence | list[pp.Sequence]) -> float:
    """Return the time a sequence, or a chain of sequences, plays, in seconds.

    Parameters
    ----------
    result : pypulseqpp.Sequence or list of pypulseqpp.Sequence
        What a sequence function returns.

    Returns
    -------
    float
        The sum of the total durations :meth:`pypulseqpp.Sequence.duration`
        reports for the sequences.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pypulseqpp import sequences
    >>> def one_delay(seconds):
    ...     seq = pp.Sequence(pp.Opts())
    ...     seq.add_block(pp.make_delay(seconds))
    ...     return seq
    >>> round(sequences.duration([one_delay(1e-3), one_delay(2e-3)]), 9)
    0.003
    """
    return float(sum(seq.duration()[0] for seq in _chain(result)))

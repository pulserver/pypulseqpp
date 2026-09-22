"""Reading a Pulseq file into a sequence that carries its system, and writing one out."""

from __future__ import annotations

__all__ = ["read", "write"]

import os as _os

import pypulseqpp as _pp
from pypulseqpp import safety as _safety

#: Definitions a file may state its design limits in, and the ``Opts``
#: argument each one supplies. A file written by this package carries none of
#: them, and the values are read in the base units of a ``.seq`` file: Hz/m
#: for a gradient amplitude, Hz/m/s for a slew rate and tesla for the field.
_LIMITS = {
    "MaxGrad": "max_grad",
    "MaxSlew": "max_slew",
    "B0": "B0",
}


def _stated(seq, name: str) -> float | None:
    """Return a single-valued definition, or None when the file has none."""
    value = seq.get_definition(name)
    if value == "" or value is None:
        return None
    return float(value[0] if isinstance(value, list) else value)


def read(
    file_path: str | _os.PathLike[str],
    *,
    max_grad: float | None = None,
    max_slew: float | None = None,
    detect_rf_use: bool = False,
    remove_duplicates: bool = True,
    verify: bool = False,
):
    """Read a Pulseq file into a sequence whose system describes what is in it.

    :meth:`~pypulseqpp.Sequence.read` replaces a sequence's contents and takes
    its rasters from the file, but leaves the sequence on whatever system it
    was constructed with. The limits and rasters of that system are what the
    design helpers solve against and what the checks in
    :mod:`pypulseqpp.safety` compare a waveform with, so a file read onto the
    shared default system is measured against limits that have nothing to do
    with it. This builds the system from the file instead.

    The rasters always come from the file. A limit or a field strength comes
    from the file when it states one, as :data:`_LIMITS` names the
    definitions, and from the argument here when one is given, which wins. A
    gradient limit that neither supplies is the larger of the
    :class:`~pypulseqpp.Opts` default and the largest amplitude the file
    actually reaches, so that the sequence read back satisfies the checks it
    was written under. Dead times are not recorded in a ``.seq`` file and stay
    at their defaults.

    Text and binary are told apart by what is in the bytes.

    Parameters
    ----------
    file_path : str | os.PathLike[str]
        The file to read.
    max_grad : float, default=None
        Gradient amplitude limit (Hz/m) for the system built. ``None`` takes
        the file's own, or the larger of the default and what the file reaches.
    max_slew : float, default=None
        Slew-rate limit (Hz/m/s), resolved the same way.
    detect_rf_use : bool, default=False
        Work out what each unlabelled pulse is for, from what it does. A file
        written before revision 1.5.0 has nowhere to record it.
    remove_duplicates : bool, default=True
        Collapse identical library rows after reading.
    verify : bool, default=False
        Check the file against the signature it carries.

    Returns
    -------
    pypulseqpp.Sequence
        The sequence, on a system built from the file.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    RuntimeError
        If the file cannot be parsed, or ``verify`` is set and the signature
        does not match.

    See Also
    --------
    write : The other direction.
    pypulseqpp.Sequence.read : Read into an existing sequence, keeping its system.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> import pypulseqpp as pp
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "gre.seq"
    >>> system = pp.Opts(max_grad=80.0, grad_unit="mT/m")
    >>> seq = pp.Sequence(system)
    >>> seq.add_block(pp.make_trapezoid("x", area=4000, duration=2e-3, system=system))
    1
    >>> _ = pp.io.write(seq, path)

    The file states no limit, so the one built admits what the file reaches,
    which is more than the default system allows:

    >>> loaded = pp.io.read(path)
    >>> loaded.num_blocks, loaded.system.max_grad > pp.Opts().max_grad
    (1, True)
    """
    seq = _pp.Sequence()
    seq.read(
        file_path,
        detect_rf_use=detect_rf_use,
        remove_duplicates=remove_duplicates,
        verify=verify,
    )

    stated = {
        argument: value
        for name, argument in _LIMITS.items()
        if (value := _stated(seq, name)) is not None
    }
    if max_grad is not None:
        stated["max_grad"] = float(max_grad)
    if max_slew is not None:
        stated["max_slew"] = float(max_slew)

    default = _pp.Opts()
    if "max_grad" not in stated:
        reached = _safety.check_max_grad(seq, default)[1].per_axis.value
        stated["max_grad"] = max(default.max_grad, float(reached))
    if "max_slew" not in stated:
        reached = _safety.check_max_slew(seq, default)[1].per_axis.value
        stated["max_slew"] = max(default.max_slew, float(reached))

    seq.system = _pp.Opts(
        rf_raster_time=seq.rf_raster_time,
        grad_raster_time=seq.grad_raster_time,
        adc_raster_time=seq.adc_raster_time,
        block_duration_raster=seq.block_duration_raster,
        **stated,
    )
    return seq


def write(
    seq,
    file_path: str | _os.PathLike[str],
    *,
    binary: bool = False,
    create_signature: bool = True,
    remove_duplicates: bool = True,
    check_timing: bool = True,
) -> str | None:
    """Write a sequence as Pulseq text or binary.

    Parameters
    ----------
    seq : pypulseqpp.Sequence
        Sequence to write; not modified.
    file_path : str | os.PathLike[str]
        Destination path.
    binary : bool, default=False
        Write the binary form instead of text. The binary writer runs no
        timing check.
    create_signature : bool, default=True
        Sign the file, in both forms.
    remove_duplicates : bool, default=True
        Write a deduplicated copy, leaving ``seq`` as it is.
    check_timing : bool, default=True
        Text only: check the timing while writing, which warns about what
        fails and records ``TotalDuration``.

    Returns
    -------
    str or None
        The text signature, or None for binary, whose signature is in the
        file rather than returned.

    See Also
    --------
    read : The other direction.
    pypulseqpp.Sequence.write : The text writer this calls.
    pypulseqpp.Sequence.write_binary : The binary writer this calls.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> import pypulseqpp as pp
    >>> directory = pathlib.Path(tempfile.mkdtemp())
    >>> seq = pp.Sequence(pp.Opts())
    >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
    1
    >>> isinstance(pp.io.write(seq, directory / "gre.seq"), str)
    True
    >>> pp.io.write(seq, directory / "gre.bin", binary=True) is None
    True
    """
    written = seq.remove_duplicates() if remove_duplicates else seq
    if binary:
        written.write_binary(str(file_path), create_signature=create_signature)
        return None
    return written.write(
        str(file_path),
        create_signature=create_signature,
        remove_duplicates=False,
        check_timing=check_timing,
    )

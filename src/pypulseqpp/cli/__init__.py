"""Command-line interfaces derived from sequence-function signatures and NumPy docstrings."""

from __future__ import annotations

__all__ = ["run"]

import argparse as _argparse
import inspect as _inspect
import re as _re
import typing as _typing

import pypulseqpp as _pp
from pypulseqpp._prescription import documented as _documented
from pypulseqpp._prescription import scalar as _scalar

#: What :func:`run` answers for itself rather than deriving a flag from. The
#: first four are the calling convention a sequence script is written to --
#: PyPulseq's own -- and ``system`` is built from the limit options.
_RESERVED = ("plot", "test_report", "write_seq", "seq_filename", "system")


def _described(doc: str | None) -> dict[str, str]:
    """Return one sentence of help per parameter: its description's first sentence."""
    return {
        name: _re.split(r"(?<=\.)\s", text, maxsplit=1)[0]
        for name, (_, text) in _documented(doc).items()
    }


def _add(parser, name: str, kind: type, default, help_text: str) -> None:
    flag = "--" + name.replace("_", "-")
    if kind is bool:
        if default is True:
            parser.add_argument(
                "--no-" + name.replace("_", "-"),
                dest=name,
                action="store_false",
                default=None,
                help=help_text or f"turn off {name}",
            )
        else:
            parser.add_argument(flag, action="store_true", default=None, help=help_text)
        return
    parser.add_argument(flag, type=kind, default=None, help=help_text)


def run(
    main,
    argv: list[str] | None = None,
    *,
    description: str | None = None,
    default_output: str = "sequence.seq",
) -> int:
    """Run a sequence script's ``main`` from the command line.

    Every keyword parameter of ``main`` annotated with a scalar becomes an
    option, described by what ``main``'s own docstring says about it. Five
    names are answered here instead: ``plot``, ``test_report``, ``write_seq``
    and ``seq_filename``, which are the calling convention rather than the
    prescription, and ``system``, which is built from the limit options.

    Parameters
    ----------
    main : callable
        The script's entry point. Returns the sequence.
    argv : list of str, default=None
        The arguments, without the program name. ``sys.argv[1:]`` when
        omitted.
    description : str, default=None
        What ``--help`` says the script does. The first line of ``main``'s
        docstring when omitted.
    default_output : str, default='sequence.seq'
        Where to write when ``--output`` is not given.

    Returns
    -------
    int
        The exit status: zero when the sequence was written.

    Notes
    -----
    The sequence ``main`` returns is written to ``--output`` with
    :func:`pypulseqpp.io.write`, in binary form under ``--binary``. The
    ``main`` of a :class:`~pypulseqpp.sequences.SequenceApp` subclass instead
    writes the application's chain of prescan and main-sequence files through
    :meth:`~pypulseqpp.sequences.SequenceApp.write`.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pypulseqpp.cli import run
    >>> def main(write_seq: bool = False, seq_filename: str = "x.seq",
    ...          *, n_x: int = 128, tr: float | None = None) -> pp.Sequence:
    ...     '''One line.
    ...
    ...     Parameters
    ...     ----------
    ...     n_x : int, optional
    ...         Readout samples.
    ...     tr : float or None, optional
    ...         Repetition time, in seconds.
    ...     '''
    ...     return pp.Sequence()

    The options are the function's parameters, described by its docstring:

    >>> import contextlib, io
    >>> text = io.StringIO()
    >>> with contextlib.redirect_stdout(text), contextlib.suppress(SystemExit):
    ...     run(main, ["--help"])
    >>> "--n-x" in text.getvalue(), "Readout samples." in text.getvalue()
    (True, True)
    """
    signature = _inspect.signature(main)
    try:
        hints = _typing.get_type_hints(main)
    except (NameError, TypeError):  # an annotation naming something not imported
        hints = {}

    parser = _argparse.ArgumentParser(
        description=description or (_inspect.getdoc(main) or "").split("\n")[0]
    )
    parser.add_argument(
        "-o", "--output", default=default_output, help="where to write the .seq"
    )
    parser.add_argument(
        "--report", action="store_true", help="print the sequence's test report"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="write the scanner's binary form rather than signed text",
    )
    parser.add_argument(
        "--max-grad-mtm", type=float, help="gradient amplitude ceiling [mT/m]"
    )
    parser.add_argument("--max-slew-tm-s", type=float, help="slew ceiling [T/m/s]")

    described = _described(_inspect.getdoc(main))
    derived: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in _RESERVED or parameter.kind in (
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        ):
            continue
        kind = _scalar(hints.get(name, parameter.annotation))
        if kind is None:
            continue
        _add(parser, name, kind, parameter.default, described.get(name, ""))
        derived.append(name)

    args = parser.parse_args(argv)

    limits = {}
    if args.max_grad_mtm is not None:
        limits.update(max_grad=args.max_grad_mtm, grad_unit="mT/m")
    if args.max_slew_tm_s is not None:
        limits.update(max_slew=args.max_slew_tm_s, slew_unit="T/m/s")

    kwargs = {name: getattr(args, name) for name in derived}
    kwargs = {name: value for name, value in kwargs.items() if value is not None}
    if "system" in signature.parameters:
        kwargs["system"] = _pp.Opts(**limits) if limits else None
    if "test_report" in signature.parameters:
        kwargs["test_report"] = args.report

    # An application writes its own chain of linked files.
    write_to = getattr(main, "write_to", None)
    if write_to is not None:
        for path in write_to(args.output, offline=not args.binary, **kwargs):
            print(f"Wrote sequence: {path}")
        return 0

    seq = main(**kwargs)
    _pp.io.write(seq, args.output, binary=args.binary)
    print(f"Wrote sequence: {args.output}")
    return 0

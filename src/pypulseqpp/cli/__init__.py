"""Command-line interfaces derived from sequence-function signatures and NumPy docstrings."""

from __future__ import annotations

__all__ = ["run", "write_sequence"]

import argparse as _argparse
import inspect as _inspect
import types as _types
import typing as _typing

import pypulseqpp as _pp

#: What :func:`run` answers for itself rather than deriving a flag from. The
#: first four are the calling convention a sequence script is written to --
#: PyPulseq's own -- and ``system`` is built from the limit options.
_RESERVED = ("plot", "test_report", "write_seq", "seq_filename", "system")

#: The annotations a flag can be made from. A parameter annotated with
#: anything else -- a system, an array, a callable -- is left to the caller.
_SCALARS = (bool, int, float, str)


def write_sequence(seq, output_path: str, *, offline: bool = True) -> str | None:
    """Write a deduplicated copy as Pulseq text or binary.

    Parameters
    ----------
    seq : pypulseqpp.Sequence
        Sequence to write; not modified.
    output_path : str
        Destination path.
    offline : bool, default True
        True writes signed text with timing warnings. False writes binary
        using write_binary's default signature setting, without a timing check.

    Returns
    -------
    str or None
        Text signature when offline; None for binary, even if the binary file
        contains a signature.
    """
    seq = seq.remove_duplicates()
    if not offline:
        seq.write_binary(output_path)
        return None
    return seq.write(output_path, check_timing=True)


def _scalar(annotation) -> type | None:
    """Return the first supported scalar type in an annotation or union."""
    if annotation is _inspect.Parameter.empty:
        return None
    members = (
        _typing.get_args(annotation)
        if _typing.get_origin(annotation) in (_typing.Union, _types.UnionType)
        else (annotation,)
    )
    return next((member for member in members if member in _SCALARS), None)


def _described(doc: str | None) -> dict[str, str]:
    """One line of help per parameter, read off the function's own docstring.

    A NumPy ``Parameters`` block states each name, then its description
    indented under it; several names sharing a description are comma
    separated, and a long list of them wraps with a trailing backslash.
    """
    lines = _inspect.cleandoc(doc or "").splitlines()
    try:
        start = next(
            i + 2
            for i, line in enumerate(lines[:-1])
            if line.strip() == "Parameters" and set(lines[i + 1].strip()) == {"-"}
        )
    except StopIteration:
        return {}

    described: dict[str, str] = {}
    names: list[str] = []
    heading = ""
    for line in lines[start:]:
        if not line.strip():
            continue
        if line[0].isspace():
            if names and not described.get(names[0]):
                for name in names:
                    described[name] = line.strip()
            continue
        if heading and set(line.strip()) == {"-"}:
            break  # the next section's underline
        heading = line
        if line.endswith("\\"):
            names += [
                n.strip() for n in line[:-1].split(":")[0].split(",") if n.strip()
            ]
            continue
        names = [n.strip() for n in line.split(":")[0].split(",") if n.strip()]
        for name in names:
            described.setdefault(name, "")
    return {name: text for name, text in described.items() if text}


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
    argv : list of str, optional
        The arguments, without the program name. ``sys.argv[1:]`` when
        omitted.
    description : str, optional
        What ``--help`` says the script does. The first line of ``main``'s
        docstring when omitted.
    default_output : str, optional
        Where to write when ``--output`` is not given.

    Returns
    -------
    int
        The exit status: zero when the sequence was written.

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

    seq = main(**kwargs)
    write_sequence(seq, args.output, offline=not args.binary)
    print(f"Wrote sequence: {args.output}")
    return 0

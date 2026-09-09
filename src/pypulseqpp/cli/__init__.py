"""Running a sequence script from a shell.

A sequence script is a function: keyword arguments in, a
:class:`~pypulseqpp.Sequence` out. :func:`run` reads that function's own
signature and docstring and builds the command line from them, so a script
gains a `--help` listing every parameter it takes without writing the parser
down a second time -- and one that cannot fall out of step with the function,
because there is only one place the parameters are stated.

This is not part of the authoring vocabulary, so it is a subpackage rather
than a name in the main namespace: `import pypulseqpp.cli` when writing a
script, never when writing a sequence.
"""

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
    """Write a finished sequence in the form its destination reads.

    The two destinations want opposite things, and a script should not have to
    remember which is which.

    ``offline=True`` -- a bench, another toolbox, a file someone will read.
    Pulseq text, signed, with the timing checked here because nothing
    downstream will check it.

    ``offline=False`` -- straight to a scanner. The binary form, which an
    interpreter parses faster and which keeps the precision the text form
    rounds away, and no signature: the interpreter checks the timing against
    its own rasters at download, which is the pass that decides.

    Both are deduplicated first. It costs a millisecond and takes several
    times the size off the file.

    Parameters
    ----------
    seq : pypulseqpp.Sequence
        The finished sequence.
    output_path : str
        Where to write it.
    offline : bool, optional
        Which of the two forms to write.

    Returns
    -------
    str or None
        The signature, when one was written.

    Examples
    --------
    >>> import os, tempfile
    >>> import pypulseqpp as pp
    >>> from pypulseqpp.cli import write_sequence
    >>> seq = pp.Sequence(pp.Opts())
    >>> _ = seq.add_block(pp.make_block_pulse(flip_angle=0.2, duration=1e-3))
    >>> gx = pp.make_trapezoid("x", flat_area=100, flat_time=2.56e-3, rise_time=2e-4)
    >>> _ = seq.add_block(gx, pp.make_adc(num_samples=80, dwell=3.2e-5, delay=gx.rise_time))
    >>> folder = tempfile.mkdtemp()

    Text is signed, so the caller is handed a signature:

    >>> len(write_sequence(seq, os.path.join(folder, "scan.seq")))
    32

    The scanner form is binary and unsigned:

    >>> write_sequence(seq, os.path.join(folder, "scan.bin"), offline=False) is None
    True
    """
    seq = seq.remove_duplicates()
    if not offline:
        seq.write_binary(output_path)
        return None
    return seq.write(output_path, check_timing=True)


def _scalar(annotation) -> type | None:
    """Read the type a flag parses out of whatever the parameter is annotated.

    A parameter is often stated as more than one thing -- ``float | None`` for
    one that may be left to the sequence to choose, ``float | tuple`` for one
    that may be given per axis. The flag parses the scalar either way, so what
    is looked for is the first scalar in the annotation.
    """
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
    """One parameter of the function, as one option of the parser."""
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

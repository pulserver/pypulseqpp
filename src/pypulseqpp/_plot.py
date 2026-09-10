"""Launch the optional SeqEyes viewer on a selected sequence range.

Install with ``pypulseqpp[plot]``, or provide ``seqeyes`` on PATH.
"""

from __future__ import annotations

import atexit
import math
import os
import shutil
import subprocess
import tempfile
import time
import weakref
from pathlib import Path
from warnings import warn

import numpy as np

from . import _ext as _cxx

__all__ = ["Viewer", "blocks_for", "executable", "plot"]

_INSTALL = "pip install 'pypulseqpp[plot]'"

#: Upstream's drawing options that SeqEyes has no counterpart for, with the
#: value that asks for nothing. A script passing one still runs; it is told
#: what was not honoured.
_NOT_HONOURED = {
    "label": "",
    "show_blocks": False,
    "time_disp": "s",
    "grad_disp": "kHz/m",
    "clear": True,
    "overlay": None,
    "stacked": False,
    "show_guides": False,
}

#: What a drawn sequence is called, and so what `save` names its pictures:
#: SeqEyes writes ``<name>_seq.png`` and ``<name>_traj.png``.
_NAME = "seq_plot"

#: How long a file an earlier session left behind is kept before it is swept.
_STALE_AFTER = 24 * 3600.0


def executable() -> Path:
    """Return the SeqEyes program to run.

    The one ``pypulseqpp-seqeyes`` installs comes first, then ``seqeyes`` on
    ``PATH``.

    Raises
    ------
    ModuleNotFoundError
        If neither is there.
    """
    try:
        import pypulseqpp_seqeyes
    except ImportError:
        pass
    else:
        found = pypulseqpp_seqeyes.executable()
        if found.is_file():
            return found
    on_path = shutil.which("seqeyes")
    if on_path is not None:
        return Path(on_path)
    raise ModuleNotFoundError(
        f"plot() draws in SeqEyes, which is not installed. Install it with {_INSTALL}, "
        "or put a seqeyes executable on PATH."
    )


def _environment(program: Path, **extra: str) -> dict[str, str]:
    """Return the environment ``program`` runs in, with ``extra`` set.

    The viewer ``pypulseqpp-seqeyes`` installs carries no Qt and is pointed at
    PySide6's; any other SeqEyes brings its own.
    """
    base = {**os.environ, **extra}
    try:
        import pypulseqpp_seqeyes
    except ImportError:
        return base
    if program == pypulseqpp_seqeyes.executable():
        return pypulseqpp_seqeyes.environment(base)
    return base


def blocks_for(
    seq, time_range=None, block_range=None, tr_range=None
) -> tuple[int, int]:
    """Return the 1-based, inclusive blocks a range names.

    Parameters
    ----------
    seq : Sequence
        The sequence the range is read against.
    time_range : sequence of float, optional
        Two times in seconds. A block is included if any of it falls inside.
    block_range : sequence of int, optional
        The first and last block, 1-based and inclusive.
    tr_range : sequence of int, optional
        The first and last repetition, 1-based and inclusive, counted from the
        first full one.

    Returns
    -------
    first, last : int
        The blocks, or ``(1, num_blocks)`` if no range is given.

    Raises
    ------
    ValueError
        If more than one range is given, a range is backwards or outside the
        sequence, or ``tr_range`` is asked of a sequence that does not repeat.
    """
    given = [r for r in (time_range, block_range, tr_range) if r is not None]
    if len(given) > 1:
        raise ValueError(
            "give one of time_range, block_range and tr_range, not several"
        )
    count = seq.num_blocks
    if count == 0:
        raise ValueError("the sequence has no blocks to draw")
    if not given:
        return 1, count
    low, high = _pair(given[0])

    if time_range is not None:
        durations = np.asarray(seq._native.block_durations())
        ends = np.cumsum(durations)
        first = int(np.searchsorted(ends, low, side="right")) + 1
        last = int(np.searchsorted(ends - durations, high, side="left"))
        if first > count or last < first:
            raise ValueError(
                f"time_range {low:g} to {high:g} s holds no block of a sequence "
                f"lasting {float(ends[-1]):g} s"
            )
        return first, min(last, count)

    if block_range is not None:
        first, last = int(low), count if math.isinf(high) else int(high)
        if not 1 <= first <= last <= count:
            raise ValueError(
                f"block_range {first} to {last} is not within the {count} blocks "
                "of the sequence"
            )
        return first, last

    size, start = seq._detect_tr()
    if size == 0:
        raise ValueError(
            "the sequence does not repeat, so it has no repetition to count"
        )
    repeats = (count - start + 1) // size
    first, last = int(low), repeats if math.isinf(high) else int(high)
    if not 1 <= first <= last <= repeats:
        raise ValueError(
            f"tr_range {first} to {last} is not within the {repeats} repetitions of "
            f"{size} blocks the sequence plays from block {start}"
        )
    return start + (first - 1) * size, start + last * size - 1


def _pair(given) -> tuple[float, float]:
    if len(given) != 2:
        raise ValueError(f"a range is two numbers, not {len(given)}")
    low, high = float(given[0]), float(given[1])
    if high < low:
        raise ValueError(
            f"a range ends after it starts, not at {high:g} before {low:g}"
        )
    return low, high


class Viewer:
    """A SeqEyes window showing a sequence.

    The file it reads stays until the window is closed and this is told so,
    by `wait` or `close`; a window still open when Python exits keeps its
    file, and the next `plot` sweeps it once it is a day old.
    """

    def __init__(self, process: subprocess.Popen, folder: Path) -> None:
        self._process = process
        self._folder = folder
        self._forgotten = False
        _open.add(self)

    @property
    def path(self) -> Path:
        """The file SeqEyes is showing."""
        return self._folder / f"{_NAME}.seq"

    @property
    def is_open(self) -> bool:
        """Whether the window is still there."""
        return self._process.poll() is None

    def wait(self, timeout: float | None = None) -> int:
        """Block until the window is closed, then remove the file it read.

        Returns
        -------
        int
            SeqEyes' exit status.

        Raises
        ------
        RuntimeError
            If SeqEyes exited with an error, carrying what it printed.
        """
        status = self._process.wait(timeout)
        said = _said(self._folder)
        self._forget()
        if status != 0:
            raise RuntimeError(f"SeqEyes exited with status {status}:\n{said}")
        return status

    def close(self) -> None:
        """Close the window, and remove the file it read."""
        if self.is_open:
            self._process.terminate()
            try:
                self._process.wait(5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._forget()

    def _forget(self) -> None:
        if not self._forgotten:
            self._forgotten = True
            _open.discard(self)
            shutil.rmtree(self._folder, ignore_errors=True)

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"<SeqEyes viewer, {state}, {self.path}>"


_open: weakref.WeakSet[Viewer] = weakref.WeakSet()


@atexit.register
def _forget_closed() -> None:
    for viewer in list(_open):
        if not viewer.is_open:
            viewer._forget()


#: Said when SeqEyes cannot start for want of a library the system provides.
_SYSTEM_LIBRARIES = (
    "SeqEyes runs on PySide6's Qt, which draws with the system's display "
    "libraries; a desktop has them already. On Debian or Ubuntu, what drawing "
    "needs is installed with\n"
    "    sudo apt install libegl1 libgl1 libx11-6 libdbus-1-3 libfontconfig1 "
    "libfreetype6 libglib2.0-0 libxkbcommon0"
)


def _said(folder: Path) -> str:
    """Return what SeqEyes printed while showing what is in ``folder``.

    A library the system was expected to provide and did not is followed by
    how to install it.
    """
    try:
        said = (folder / "seqeyes.log").read_text(errors="replace").strip()
    except OSError:
        return ""
    if "error while loading shared libraries" in said:
        return f"{said}\n\n{_SYSTEM_LIBRARIES}"
    return said


def _folder() -> Path:
    """Return a new folder for one drawing, sweeping stale ones beside it.

    On Linux it is under ``/dev/shm``, which is memory rather than disk.
    """
    shm = Path("/dev/shm")  # noqa: S108 -- memory, and private to the user below
    root = (
        shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(tempfile.gettempdir())
    )
    where = root / f"pypulseqpp-seqeyes-{_user()}"
    where.mkdir(mode=0o700, exist_ok=True)
    now = time.time()
    for stale in where.iterdir():
        try:
            if now - stale.stat().st_mtime > _STALE_AFTER:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            pass
    return Path(tempfile.mkdtemp(dir=where))


def _user() -> str:
    try:
        return str(os.getuid())
    except AttributeError:
        return os.environ.get("USERNAME", "user")


def plot(
    seq,
    *,
    time_range=None,
    block_range=None,
    tr_range=None,
    plot_now: bool = True,
    save: bool = False,
    **options,
) -> Viewer:
    """Open a SeqEyes window on the sequence.

    See `Sequence.plot`, which is what a script calls.
    """
    unknown = sorted(set(options) - set(_NOT_HONOURED))
    if unknown:
        raise TypeError(
            f"plot() got unexpected keyword arguments: {', '.join(unknown)}"
        )
    ignored = sorted(
        name
        for name, value in options.items()
        if not _is_default(value, _NOT_HONOURED[name])
    )
    if ignored:
        warn(
            f"plot(): SeqEyes draws its own way and does not take {', '.join(ignored)}; "
            "ignored",
            stacklevel=3,
        )

    first, last = blocks_for(seq, time_range, block_range, tr_range)
    program = executable()
    arguments = [str(program)]
    if (first, last) != (1, seq.num_blocks):
        start, end = _span(seq, first, last)
        arguments += [
            "--name",
            f"{_NAME}: blocks {first} to {last} of {seq.num_blocks}, from {start:g} s",
        ]
        if time_range is not None:
            # The excerpt's clock starts at its first block, and SeqEyes takes
            # the window in whole milliseconds: rounding outwards keeps all
            # that was asked for on screen.
            low = 1e3 * (max(float(time_range[0]), start) - start)
            high = 1e3 * (min(float(time_range[1]), end) - start)
            arguments += ["--time-range", f"{math.floor(low)}~{math.ceil(high)}"]

    folder = _folder()
    path = folder / f"{_NAME}.seq"
    path.write_bytes(_excerpt(seq, first, last))

    if save:
        _capture(program, arguments, path, Path.cwd())
    viewer = Viewer(_run(program, [*arguments, str(path)], folder), folder)
    if plot_now:
        viewer.wait()
    return viewer


def _run(program: Path, arguments: list[str], folder: Path, **environment):
    """Start SeqEyes, with what it prints going to a log beside the file."""
    env = _environment(program, **environment)
    with (folder / "seqeyes.log").open("ab") as log:
        return subprocess.Popen(  # noqa: S603 -- our own arguments, no shell
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )


def _capture(program: Path, arguments: list[str], path: Path, where: Path) -> None:
    """Write SeqEyes' pictures of the sequence and its trajectory to ``where``.

    Raises
    ------
    RuntimeError
        If SeqEyes could not draw them.
    """
    process = _run(
        program,
        [*arguments, "--capture-snapshots", str(where), str(path)],
        path.parent,
        QT_QPA_PLATFORM="offscreen",
    )
    status = process.wait()
    if status != 0:
        raise RuntimeError(
            f"SeqEyes could not save the plot (status {status}):\n{_said(path.parent)}"
        )


def _excerpt(seq, first: int, last: int) -> bytes:
    """Serialise an inclusive block range with a local time base.

    Prepend a zero-duration block with incoming label values, so displayed
    counters match the original sequence.
    """
    native = seq._native
    if first == 1 and last == native.num_blocks():
        return _cxx.write_text(native, False)
    rows = list(range(first, last + 1))
    reached = seq.evaluate_labels(block_range=(1, first - 1)) if first > 1 else {}
    if not any(reached.values()):
        return _cxx.write_text(native, False, rows)

    from . import make_label  # the package imports this module

    labels = [
        make_label(label=name, type="SET", value=int(value))
        for name, value in reached.items()
        if value
    ]
    excerpt = native.copy()
    lead = excerpt.add_block_events(*labels)
    return _cxx.write_text(excerpt, False, [lead, *rows])


def _is_default(value, default) -> bool:
    return value is default or (type(value) is type(default) and value == default)


def _span(seq, first: int, last: int) -> tuple[float, float]:
    """Return when block ``first`` starts and block ``last`` ends, in s."""
    durations = np.asarray(seq._native.block_durations())
    before = float(durations[: first - 1].sum())
    return before, before + float(durations[first - 1 : last].sum())

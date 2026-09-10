"""SeqEyes, the Pulseq sequence viewer, built to run on the Qt PySide6 installs.

The program is ``bin/seqeyes`` beside this file. It is linked against Qt but
carries none: it runs on the Qt that ``PySide6-Essentials`` installs, which
this package depends on. `environment` says where that is, and `main` runs the
program with it, which is what the ``pypulseqpp-seqeyes`` command does.

This package is GPL-3.0-or-later. SeqEyes itself is BSD-3-Clause; QCustomPlot,
the plotting widget compiled into it, is GPL.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

__all__ = ["environment", "executable", "main"]


def executable() -> Path:
    """Return the SeqEyes program this package installs."""
    name = "seqeyes.exe" if sys.platform == "win32" else "seqeyes"
    return Path(__file__).parent / "bin" / name


def _qt() -> tuple[Path, Path]:
    """Return where PySide6 keeps Qt's libraries, and its plugins."""
    spec = importlib.util.find_spec("PySide6")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "SeqEyes runs on the Qt that PySide6-Essentials installs, and it is "
            "missing. Reinstall pypulseqpp-seqeyes."
        )
    root = Path(next(iter(spec.submodule_search_locations)))
    if sys.platform == "win32":
        return root, root / "plugins"
    return root / "Qt" / "lib", root / "Qt" / "plugins"


def environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment SeqEyes runs in: ``base``, pointed at PySide6's Qt.

    Parameters
    ----------
    base : mapping, optional
        The environment to start from; the current one by default.

    Returns
    -------
    dict
        ``base`` with Qt's plugins named, and on Windows and Linux its
        libraries put first on the search path. On macOS the program's own
        run path finds them.
    """
    env = dict(os.environ if base is None else base)
    libraries, plugins = _qt()
    if sys.platform == "win32":
        env["PATH"] = os.pathsep.join(filter(None, [str(libraries), env.get("PATH")]))
    elif sys.platform != "darwin":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(libraries), env.get("LD_LIBRARY_PATH")])
        )
    env["QT_PLUGIN_PATH"] = str(plugins)
    return env


def main(argv: list[str] | None = None) -> int:
    """Run SeqEyes with ``argv`` -- a file to open and its options -- and wait."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    return subprocess.call([str(executable()), *arguments], env=environment())  # noqa: S603

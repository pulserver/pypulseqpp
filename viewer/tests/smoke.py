"""Open a sequence in the installed viewer, headless, and draw it.

Run by cibuildwheel against the built wheel, in a fresh environment that has
only it and what it depends on -- so it holds that the wheel runs on the Qt
PySide6 installs, on the oldest system its tag admits.

``gre.seq`` is ``write_gre.seq`` from the MIT-licensed reference corpus of
pypulseq-matlab-like, as pypulseqpp's own tests carry it.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pypulseqpp_seqeyes as viewer

program = str(viewer.executable())
env = viewer.environment({**os.environ, "QT_QPA_PLATFORM": "offscreen"})


def run(*arguments: str) -> subprocess.CompletedProcess:
    """Run SeqEyes with ``arguments``, print what it said, and stop if it failed."""
    result = subprocess.run(  # noqa: S603
        [program, *arguments], env=env, capture_output=True, text=True, timeout=600
    )
    said = (result.stdout + result.stderr).strip()
    print(f"$ seqeyes {' '.join(arguments)}\n{said}\n(status {result.returncode})")
    if result.returncode != 0:
        sys.exit(f"seqeyes {arguments[0]} failed")
    return result


run("--version")
with tempfile.TemporaryDirectory() as out:
    run("--capture-snapshots", out, str(Path(__file__).with_name("gre.seq")))
    drawn = sorted(path.name for path in Path(out).glob("*.png"))
    print("drew", drawn)
    if drawn != ["gre_seq.png", "gre_traj.png"]:
        sys.exit("seqeyes did not draw the sequence and its trajectory")

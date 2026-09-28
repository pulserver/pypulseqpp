"""Design, write, read back and simulate a sequence in Pyodide, where the core runs one thread.

Run by the Pyodide job of the test workflow, in the interpreter cibuildwheel
builds the wheel for; every path that divides work among threads elsewhere
runs here.
"""

import tempfile
from pathlib import Path

import numpy as np

import pypulseqpp as pp
from pypulseqpp import sequences

seq = sequences.gre2D_sequence(n_x=32, n_y=8, n_slices=1, tr=None, n_dummy=0)
ok, report = seq.check_timing()
if not ok:
    raise SystemExit(f"the timing check failed: {report}")
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "gre2d.seq"
    seq.write(str(path))
    again = pp.Sequence()
    again.read(str(path))
if again.duration()[0] != seq.duration()[0]:
    raise SystemExit("the sequence read back lasts another time")

positions = np.random.default_rng(0).uniform(-0.1, 0.1, (256, 3))
spins = pp.Isochromats(positions, t1=1.0, t2=0.1, receive=np.ones((256, 2)))
signal = seq.simulate(spins)
if signal.shape[0] != 2 or not np.all(np.isfinite(signal)):
    raise SystemExit(f"the simulation returned {signal.shape} with non-finite samples")
print(f"pypulseqpp {pp.__version__}: {signal.shape[1]} samples on 2 coils")

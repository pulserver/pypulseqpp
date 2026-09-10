# pypulseqpp-seqeyes

[SeqEyes](https://github.com/xingwangyong/seqeyes), the Pulseq sequence viewer,
built for `pypulseqpp`'s `Sequence.plot`. It is installed with
`pip install 'pypulseqpp[plot]'` and is not needed for anything else.

The wheel carries the SeqEyes program and no Qt: it runs on the Qt that
`PySide6-Essentials` installs. `pypulseqpp-seqeyes file.seq` opens a file
from the command line, with SeqEyes' own options.

## Licence

GPL-3.0-or-later. SeqEyes is BSD-3-Clause (`seqeyes/LICENSE`, and
`seqeyes/licenses/license_safe_pns` for its PNS model); QCustomPlot, the
plotting widget compiled into it, is GPL-3.0-or-later, so the program they
make is too. `pypulseqpp` itself is MIT and does not depend on this package.

## Building

SeqEyes is the submodule in `seqeyes/`, pinned to a release and built as it
stands. A build needs Qt 6.8 or later, found through `CMAKE_PREFIX_PATH`:

```bash
git submodule update --init --recursive
CMAKE_PREFIX_PATH=/path/to/Qt/6.8.3/gcc_64 pip install ./viewer
```

The release wheels are built by `cibuildwheel`, configured in `pyproject.toml`.

# pypulseqpp-seqeyes

[SeqEyes](https://github.com/xingwangyong/seqeyes), the Pulseq sequence viewer,
built for `pypulseqpp`'s `Sequence.plot`. It is installed with
`pip install 'pypulseqpp[plot]'` and is not needed for anything else.

The wheel carries the SeqEyes program and no Qt: it runs on the Qt that
`PySide6-Essentials` installs. `pypulseqpp-seqeyes file.seq` opens a file
from the command line, with SeqEyes' own options.

## Linux

PySide6's Qt draws with the system's display libraries. A desktop has them;
a bare server or container needs, on Debian or Ubuntu,

```bash
# For offscreen rendering with QT_QPA_PLATFORM=offscreen
sudo apt install libegl1 libgl1 libx11-6 libdbus-1-3 libfontconfig1 \
  libfreetype6 libglib2.0-0 libxkbcommon0
# And to open a window under X11 (Wayland needs none of these)
sudo apt install libx11-xcb1 libxkbcommon-x11-0 libxcb1 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-render0 libxcb-shape0 libxcb-shm0 \
  libxcb-sync1 libxcb-util1 libxcb-xfixes0 libxcb-xkb1
```

`libxcb-cursor0` is the one an Ubuntu desktop may lack: Qt needs it to open an
X11 window, and says so if it is missing.

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

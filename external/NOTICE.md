# What is under `external/`

## MRArbGrad

`MRArbGrad/` is a submodule of <https://github.com/mcencini/MRArbGrad>, a fork
of <https://github.com/RyanShanghaitech/MRArbGrad>. It solves the problem this
package has no other answer to: given a k-space path stated as geometry, find
the gradient that traces it in least time without leaving the amplitude and
slew limits anywhere along it.

> Luo R, Huang H, Miao Q, Xu J, Hu P, Qi H. "Real-Time Gradient Waveform Design
> for Arbitrary k-Space Trajectories." IEEE Transactions on Biomedical
> Engineering. 2026;1-12.

It is MIT licensed; the text travels with the submodule, in
`MRArbGrad/LICENSE`.

**A submodule rather than a copy**, and a fork rather than upstream. A
submodule is pinned to one commit, so what a wheel compiles is exactly what
the last person to bump it chose; the fork is so that the commit stays
reachable whatever happens to upstream. Bumping it is
`git -C external/MRArbGrad checkout <ref>` followed by a commit here, and
`tests/test_arbgrad_spiral.py` says whether the waveforms moved.

**Three of its files are compiled**, named in `CMakeLists.txt`: `mag/Mag.cpp`,
`utility/global.cpp` and `utility/v3.cpp`. The rest of the solver is headers,
pulled in by `src/cpp/bindings/arbgrad.cpp`, which is this package's own
pybind11 binding over it. Its own CPython bindings (`ext/main.cpp`) and its
Python package are not built: the entry points here are
`pypulseqpp._ext.arbgrad`, and the shot ordering and rotation that a scan
replays a base arm under are Python policy rather than the solver's.

**A constant-pitch spiral is not a separate trajectory.** `VDSpiral_TrajFunc`
reduces algebraically to `rho = kRhoPhi * phi` when its two shape parameters
are equal, so `traj/Spiral.h` would be a redundant special case and no binding
is offered for it. `pypulseqpp.make_spiral`-style callers pass equal
parameters instead.

## The Poisson-disc kernel

`src/cpp/bindings/sampling.cpp` is not from here. The variable-density
Poisson-disc behaviour was reimplemented in C++ against SigPy's
`sigpy.mri.samp.poisson` as a reference; SigPy is BSD 3-Clause,
<https://github.com/mikgroup/sigpy>.

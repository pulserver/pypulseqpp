# License and third-party notices

The pypulseqpp core is distributed under the [MIT licence](https://github.com/pulserver/pypulseqpp/blob/main/LICENSE).
Vendored or transcribed third-party components retain their own licences:

| Component | Licence | Notice |
| --- | --- | --- |
| SAFE PNS prediction implementation | BSD 3-Clause | [`LICENSES/safe_pns_prediction-BSD-3-Clause.txt`](https://github.com/pulserver/pypulseqpp/blob/main/LICENSES/safe_pns_prediction-BSD-3-Clause.txt) |
| pocketfft | BSD 3-Clause | [`LICENSES/pocketfft-BSD-3-Clause.txt`](https://github.com/pulserver/pypulseqpp/blob/main/LICENSES/pocketfft-BSD-3-Clause.txt) |
| SigPy-derived routines | BSD 3-Clause | [`LICENSES/SigPy-BSD-3-Clause.txt`](https://github.com/pulserver/pypulseqpp/blob/main/LICENSES/SigPy-BSD-3-Clause.txt) |
| [MRArbGrad](https://github.com/mcencini/MRArbGrad) | MIT | [`external/NOTICE.md`](https://github.com/pulserver/pypulseqpp/blob/main/external/NOTICE.md), and [`LICENSE`](https://github.com/mcencini/MRArbGrad/blob/main/LICENSE) in the submodule |
| [SeqEyes](https://github.com/xingwangyong/seqeyes) | BSD 3-Clause | [`LICENSE`](https://github.com/xingwangyong/seqeyes/blob/main/LICENSE) in the submodule |

MRArbGrad and SeqEyes are Git submodules, under `external/` and `viewer/`, so
their licence texts are part of the checkout rather than of this repository's
own tree. [`external/NOTICE.md`](https://github.com/pulserver/pypulseqpp/blob/main/external/NOTICE.md)
states which of MRArbGrad's files are compiled and cites the method.

The viewer is packaged separately as `pypulseqpp-seqeyes` under
[GPL-3.0-or-later](https://github.com/pulserver/pypulseqpp/blob/main/viewer/LICENSE),
which covers the package built around SeqEyes rather than SeqEyes itself.
Installing the optional viewer does not change the MIT licence of the core
package.

The pypulseqpp logo derives from the wordmark and bipolar-gradient mark of the
MIT-licensed [PyPulseq](https://github.com/imr-framework/pypulseq) project,
redrawn here as SVG and extended with the `++` suffix. The SVG artwork in this
repository is maintained as pypulseqpp source material.

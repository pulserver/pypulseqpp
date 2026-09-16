# Related projects

pypulseqpp sits between the Pulseq ecosystem and the Pulserver execution stack.

| Project | Relationship |
| --- | --- |
| [Pulseq](https://github.com/pulseq/pulseq) | The open sequence file format and the MATLAB toolbox that defines it. pypulseqpp reads and writes Pulseq 1.2 to 1.5 text and binary files. |
| [PyPulseq](https://github.com/imr-framework/pypulseq) | The Python interface pypulseqpp is compatible with, and a runtime dependency. Supported upstream signatures, event conventions and reference output are preserved; the event factories are re-exported. |
| `pypulseq-matlab-like` | A transcription of the MATLAB Pulseq toolbox, used as the file-format authority in the tests. It is a test-only dependency named by URL; the tests that need it skip when it is absent. |
| [Pulserver](https://github.com/pulserver/pulserver) | Scanner execution, segmentation, protocol contracts and consoles. Vendor-specific execution logic lives there, not here. |
| [SigPy](https://github.com/mikgroup/sigpy) | Source of the SLR, adiabatic, B1-selective, multiband-phase and Poisson-disc algorithms transcribed here under BSD-3-Clause. pypulseqpp does not depend on it at runtime. |
| [MRArbGrad](https://github.com/RyanShanghaitech/MRArbGrad) | The arbitrary-gradient solver behind the spiral, rosette and trajectory-to-gradient designs, vendored as a submodule of the [fork](https://github.com/mcencini/MRArbGrad) this package pins. |
| [mrsd](https://pypi.org/project/mrsd/) | Draws the publication diagram behind {func}`pypulseqpp.plot.paper_plot`. |
| [SeqEyes](https://github.com/xingwangyong/seqeyes) | The optional sequence viewer behind {func}`pypulseqpp.plot.plot`, packaged separately as the GPL-licensed `pypulseqpp-seqeyes`. |

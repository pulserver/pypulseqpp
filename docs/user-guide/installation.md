# Installation

Install the core package from PyPI:

```bash
pip install pypulseqpp
```

Optional facilities are installed as extras:

| Extra | Command | Purpose |
| --- | --- | --- |
| SeqEyes viewer | `pip install 'pypulseqpp[plot]'` | Interactive sequence viewing through the separately distributed GPL viewer. |
| Intel MKL | `pip install 'pypulseqpp[mkl]'` | Optional FFT backend for mechanical-resonance analysis on x86-64. |
| FSE design | `pip install 'pypulseqpp[design]'` | `torchsim`, used by optimized fast-spin-echo refocusing schedules. |

Developer installation is documented in {doc}`../developer-guide/installation`.

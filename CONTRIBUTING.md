# Contributing to pypulseqpp

The [Developer Guide](https://pulserver.github.io/pypulseqpp/latest/developer-guide/index.html)
documents the toolchain, editable installation, coding and documentation
conventions, pre-commit hooks and pull-request workflow.

A quick development setup is:

```bash
git clone --recurse-submodules https://github.com/YOUR-USER/pypulseqpp.git
cd pypulseqpp
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,examples]' -r tests/requirements.txt
pre-commit install
```

Before opening a pull request:

```bash
bash scripts/format_and_lint.sh --check
pytest -q
```

Open the pull request against [`pulserver/pypulseqpp:main`](https://github.com/pulserver/pypulseqpp/compare).
Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

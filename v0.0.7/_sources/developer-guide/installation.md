# Editable installation

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -r tests/requirements.txt
```

The `dev` extra contains lint, test and documentation tools. Gallery builds
also require the `examples` extra because optimized FSE examples use
`torchsim`:

```bash
python -m pip install -e '.[dev,examples]' -r tests/requirements.txt
```

# Coding style

Ruff supplies Python formatting and linting; configuration and the pinned
version are in `pyproject.toml`. Run:

```bash
bash scripts/format_and_lint.sh
pytest -q
```

Use pytest functions and fixtures rather than `unittest.TestCase`. Test names
state the invariant they protect. Numerical or file-format changes require
parity or regression coverage appropriate to the affected contract.

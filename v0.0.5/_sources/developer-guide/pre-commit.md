# Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

A deliberate one-off bypass uses `git commit --no-verify`; one hook can be
skipped with `SKIP=<hook-id> git commit`. A bypass does not bypass CI and must
not be used to conceal a failing check.

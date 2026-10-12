# Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

The format-and-lint hook runs `scripts/format_and_lint.sh`, the script the
Style workflow runs with `--check`, so a commit that passes locally passes the
style check in CI. The hooks skip the submodules under `external/` and
`viewer/seqeyes/`.

A deliberate one-off bypass uses `git commit --no-verify`; one hook can be
skipped with `SKIP=<hook-id> git commit`. A bypass does not bypass CI and must
not be used to conceal a failing check.

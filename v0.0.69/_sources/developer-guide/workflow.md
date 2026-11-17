# Workflow

## Branches

1. Fork `pulserver/pypulseqpp` on GitHub and clone the fork with submodules.
2. Add `https://github.com/pulserver/pypulseqpp.git` as the `upstream` remote.
3. Update local `main`, then create a focused feature branch.
4. Commit tested changes and push the branch to the fork.
5. Open a pull request against `pulserver/pypulseqpp:main`.

Keep commits reviewable and avoid unrelated formatting or prose churn.

## Pull requests

A pull request follows the repository template: *What changed*, *Why*, with
any linked issue or discussion, *Validation*, with the exact commands run and
their results, and *Documentation impact*. New behaviour requires a regression
test where practical.

## Continuous integration

A pull request runs the Tests workflow (Python 3.14 on Linux, macOS and
Windows, Python 3.10 on Linux, the plotting tests and a sanitizer build) and
the Style workflow. The documentation is built on a pull request only when a
maintainer comments `/build-docs`; `main` builds and publishes it after the
merge.

## Code of conduct

The canonical participation policy is the root
[`CODE_OF_CONDUCT.md`](https://github.com/pulserver/pypulseqpp/blob/main/CODE_OF_CONDUCT.md).
All project spaces, including issues, discussions and pull requests, follow
that policy.

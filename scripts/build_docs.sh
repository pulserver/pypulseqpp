#!/usr/bin/env bash
# Build the HTML documentation from this checkout.
#
# The package is compiled into docs/build/site and the pages are generated from
# that build, so they describe the code in the tree rather than whichever
# pypulseqpp happens to be installed. The CMake build directory is kept, so a
# rebuild recompiles only what changed.
#
#   bash scripts/build_docs.sh            # docs/build/html/index.html
#   bash scripts/build_docs.sh -E         # extra arguments go to sphinx-build
#
# Needs the documentation tools: pip install '.[doc]' (or '.[dev]').
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
site="$PWD/docs/build/site"
out="$PWD/docs/build/html"

if ! "$PYTHON_BIN" -c "import sphinx, myst_parser, sphinx_book_theme, sphinx_copybutton, linkify_it" 2>/dev/null; then
    echo "build_docs.sh: the documentation tools are missing; install them with pip install '.[doc]'" >&2
    exit 1
fi

"$PYTHON_BIN" -m pip install --quiet --no-deps --upgrade --target "$site" \
    -Cbuild-dir="$PWD/docs/build/cmake" .

# Sphinx runs without the site module (-S): the build goes first on the path
# and the interpreter's package directories after the standard library, in
# the order the interpreter itself lists them. They supply Sphinx and the
# dependencies, but their .pth files are not processed, so an editable
# install of pypulseqpp cannot redirect the import away from the build.
packages="$("$PYTHON_BIN" -c 'import os, site, sys; dirs = set(site.getsitepackages() + [site.getusersitepackages()]); print(os.pathsep.join(p for p in sys.path if p in dirs))')"
bootstrap='
import os, pathlib, runpy, sys
build, packages, action = sys.argv[1:4]
sys.path.insert(0, build)
sys.path.extend(p for p in packages.split(os.pathsep) if p)
if action == "check":
    import pypulseqpp
    found = pathlib.Path(pypulseqpp.__file__).resolve()
    if pathlib.Path(build).resolve() not in found.parents:
        sys.exit(f"build_docs.sh: imported pypulseqpp from {found}, not from {build}")
else:
    sys.argv = ["sphinx-build", *sys.argv[4:]]
    runpy.run_module("sphinx", run_name="__main__")
'

"$PYTHON_BIN" -S -c "$bootstrap" "$site" "$packages" check
"$PYTHON_BIN" -S -c "$bootstrap" "$site" "$packages" build \
    -W --keep-going -b html docs "$out" "$@"
echo "Built $out/index.html"

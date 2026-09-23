#!/usr/bin/env bash
# Build the single-page Sphinx manual and print it to docs/build/pypulseqpp-docs.pdf.
#
# Sphinx renders the sources as one HTML page, and scripts/print_pdf.py prints
# that page with headless Chromium once MathJax has typeset its equations. The
# gallery outputs of the HTML build are reused, so no script runs twice.
#
#   bash scripts/build_docs_pdf.sh          # the HTML build first, then the PDF
#   SKIP_HTML=1 bash scripts/build_docs_pdf.sh   # after build_docs.sh has run
#
# Needs the documentation tools and Playwright's Chromium:
#
#   pip install '.[doc]' && python -m playwright install chromium
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "${SKIP_HTML:-0}" != 1 ]; then
  bash scripts/build_docs.sh
fi
"$PYTHON_BIN" -m sphinx -W --keep-going -d docs/build/single-doctrees \
  -b singlehtml docs docs/build/singlehtml

version="$("$PYTHON_BIN" -c 'import importlib.metadata as m; print(m.version("pypulseqpp"))' 2>/dev/null || true)"
"$PYTHON_BIN" scripts/print_pdf.py docs/build/singlehtml docs/build/pypulseqpp-docs.pdf \
  --version "${version:+Version ${version}}"
echo "Built $PWD/docs/build/pypulseqpp-docs.pdf"

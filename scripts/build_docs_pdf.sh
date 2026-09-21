#!/usr/bin/env bash
# Build the single-page Sphinx manual and render it to PDF.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${SKIP_HTML:-0}" != 1 ]; then
  bash scripts/build_docs.sh
fi
python -m sphinx -W --keep-going -d docs/build/single-doctrees \
  -b singlehtml docs docs/build/singlehtml
cp docs/_static/pypulseqpp-logo.svg docs/_static/architecture.svg \
  docs/build/singlehtml/_static/
python - <<'PY'
from pathlib import Path

from bs4 import BeautifulSoup
from latex2mathml.converter import convert

source = Path("docs/build/singlehtml/index.html")
soup = BeautifulSoup(source.read_text(encoding="utf-8"), "html.parser")
for element in soup.select(
    "script, style, link[rel=stylesheet], header, footer, nav, aside, button, "
    "dialog, form, .skip-link, #pst-scroll-pixel-helper, .pst-async-banner-revealer"
):
    element.decompose()
for image in soup.find_all("img"):
    source_url = image.get("src", "")
    prefix = "https://raw.githubusercontent.com/pulserver/pypulseqpp/main/docs/_static/"
    if source_url.startswith(prefix):
        image["src"] = "_static/" + source_url.removeprefix(prefix)
    elif source_url.startswith(("http://", "https://")):
        image.decompose()
for equation in soup.select(".math"):
    latex = equation.get_text().strip()
    display = equation.name == "div"
    delimiters = ("\\[", "\\]") if display else ("\\(", "\\)")
    if latex.startswith(delimiters[0]) and latex.endswith(delimiters[1]):
        latex = latex[len(delimiters[0]) : -len(delimiters[1])]
    mathml = BeautifulSoup(
        convert(latex, display="block" if display else "inline"), "html.parser"
    ).math
    equation.replace_with(mathml)
targets = {element["id"] for element in soup.find_all(id=True)}
for link in soup.find_all("a", href=True):
    href = link["href"]
    if not href.startswith("#") or href[1:] in targets:
        continue
    candidate = href.rsplit("#", 1)[-1]
    if candidate in targets:
        link["href"] = "#" + candidate
    else:
        del link["href"]
cover = soup.new_tag("section", attrs={"class": "pdf-cover"})
logo = soup.new_tag("img", src="_static/pypulseqpp-logo.svg", alt="pypulseqpp")
cover.append(logo)
title = soup.new_tag("h1")
title.string = "pypulseqpp documentation"
cover.append(title)
heading = soup.new_tag("h2")
heading.string = "Contents"
cover.append(heading)
contents = soup.new_tag("ul")
for label, target in (
    ("User guide", "document-user-guide/index"),
    ("Explanations", "document-explanations/index"),
    ("Examples", "document-examples/index"),
    ("API reference", "document-api/index"),
    ("Developer guide", "document-developer-guide/index"),
    ("Miscellaneous", "document-misc/index"),
):
    item = soup.new_tag("li")
    link = soup.new_tag("a", href=f"#{target}")
    link.string = label
    item.append(link)
    contents.append(item)
cover.append(contents)
soup.body.insert(0, cover)
output = Path("docs/build/singlehtml/print.html")
output.write_text(str(soup), encoding="utf-8")
PY
weasyprint -s docs/_static/pdf.css \
  docs/build/singlehtml/print.html docs/build/pypulseqpp-docs.pdf
echo "Built $PWD/docs/build/pypulseqpp-docs.pdf"

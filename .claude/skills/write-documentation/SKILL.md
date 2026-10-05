---
name: write-documentation
description: Write, move or audit pypulseqpp documentation — API docstrings, explanation pages, gallery examples, user-guide and developer-guide pages. Use before creating or substantially modifying any documentation, and when deciding which documentation type a piece of material belongs to.
---

# Write documentation

Two documents govern documentation and both are binding. Read them before
writing:

- `docs/developer-guide/documentation.md` — what belongs in each form of
  documentation and how each is written.
- `docs/developer-guide/terminology.md` — terminology, register, units,
  frames, rasters, safety language and source-of-truth rules.

Where they differ, the terminology page governs conventions and the
documentation page governs type and register.

## Choose the type first

| Location | Type | Answers |
|---|---|---|
| `docs/user-guide/` | Task-oriented how-to | How do I install and use the project? |
| `docs/explanations/` | Conceptual explanation | Why does this work this way? |
| `gallery/` | Executable examples, built into `docs/generated/gallery/` | What does a representative scientific workflow look like? |
| `docs/api/` | Reference | What exactly does this object do? |
| `docs/sequences.md` | Catalogue of the shipped sequences | Which sequences exist, and what does one of them look like? |
| `docs/developer-guide/` | Contributor procedure and conventions | How is this repository developed? |

The types are not interchangeable, and the prose style of one does not
transfer into another. An example links to conceptual material rather than
restating it. A representative configuration of a shipped sequence, with its
diagram, is reference material and belongs on that sequence's page, not in the
gallery. A gallery example exists because running it shows something
scientifically or computationally useful; an API demonstration, a constructor
catalogue or a set of configurations whose only result is that they run does
not belong there. The Examples section has three parts, in this order:

- **Course** — `gallery/01-pulseq-basics` to `07-custom-modules`, numbered
  lessons read in order, each with a short introduction and explicit learning
  objectives; transitions between lessons are allowed. The Course is the
  shortest coherent path that gives a new user the framework's core mental
  model and enough practical competence to work independently.
- **Tours** — `gallery/08-tours`, standalone examples: useful applications,
  advanced branches or specialised workflows that are not necessary for that
  core competence. A Tour opens with its objective and a *Prerequisites* line
  naming the Course lessons it assumes, and has no previous/next links.
- **Sequence catalogue** — `gallery/10-*` to `16-*`, one executed page per
  shipped sequence, linked from `docs/sequences.md`; see the `add-a-sequence`
  skill.

The documentation guide's "Course, Tours and catalogue" section defines their
register. The Examples landing page carries one table per part, and
`tests/test_docs_navigation.py` holds the Course and Tours tables to the
gallery directories.

Explanation pages are one flat list under `docs/explanations/`, each with one
obvious purpose and, where appropriate, a conceptual anchor (figure, table or
equation); no page is nested below another.

## Mechanics

A gallery script is a `.py` file whose module docstring is the page's title and
opening; `# %%` starts a text cell, and code a reader would not type goes
between `# sphinx_gallery_start_ignore` and `# sphinx_gallery_end_ignore`.
Every script is executed at build time.

`gallery/` is flat — sphinx-gallery reads one level of subdirectories — with
one directory per landing page, listed in `GALLERY_SECTIONS` in `docs/conf.py`.
The navigable hierarchy is built by the pages under `docs/examples/`, each
carrying a table and a hidden toctree over the same entries.

Every example page carries an *Open in Colab* badge under its title, inserted
at build time by `docs/colab.py`, which also writes a copy of each gallery
notebook into the built site under `_colab/` with a note and a `%pip install`
cell in front; the notebook the page offers for download is left as
sphinx-gallery writes it. A section whose scripts import more than
`pypulseqpp[plot]` and matplotlib lists the packages in `SECTION_PACKAGES`.
An explanation page with more than one `##` section opens with a TL;DR
admonition (```` ```{admonition} TL;DR ```` with `:class: tldr`) directly
under its title; a single-section page may omit it, and landing, API and
example pages have none. `tests/test_docs_explanations.py` holds both
conventions.

Figures take their typography, resolution and layout from
`docs/figure_style.py`; no gallery script sets font sizes, DPI or `rcParams`.
The gallery's image scraper narrows a figure wider than the column, so a
script draws at the size its content needs. A legend describing several
panels goes outside them, with `layout="constrained"` and
`fig.legend(loc="outside upper center")`. A static diagram is a light/dark SVG
pair shown with the `only-light` and `only-dark` classes. Public plotting
defaults, such as canvas sizes, are not changed for the documentation.

Three generators run on `builder-inited` and write into `docs/generated/`,
which is not tracked: `docs/explanation_figures.py`,
`docs/sequence_reference.py` and `docs/api_objects.py`.

Snippets in `docs/user-guide/*.md` are executed as doctests by
`tests/test_docs_guides.py`, so a procedure that cannot be followed literally
fails the suite.

## Verify, then build

Verify substantive semantics against the implementation, the tests, the `.seq`
format authority (`pypulseq-matlab-like`), upstream PyPulseq, and the Pulseq
specification or primary literature — in that order. Existing prose is not
evidence. Flag an unresolved discrepancy rather than guessing.

When revising, preserve technically good material, move misplaced material to
the correct type rather than deleting it, and avoid unrelated stylistic churn.
Check explicitly for conversational prose, paraphrases of established
terminology, personification, taglines, code narration and imprecise
accessibility.

This package computes checks and estimates. A passing check does not establish
that a sequence is safe to run on a scanner or on a subject. Never write
"safe", "validated", "compliant" or "approved" of a sequence that passed one,
and preserve the existing disclaimers verbatim.

Then build and inspect the result, including the sidebar hierarchy:

```bash
bash scripts/build_docs.sh          # docs/build/html/index.html
bash scripts/build_docs_pdf.sh      # docs/build/pypulseqpp-docs.pdf
pytest -q tests/test_docs_guides.py tests/test_docs_sequences.py \
          tests/test_docs_reference.py tests/test_docs_explanations.py \
          tests/test_docstrings.py tests/test_docstring_defaults.py
```

The build runs under `-W`, so a broken reference or an unreachable page fails
it. The PDF is the single-page build printed by headless Chromium once MathJax
has typeset it, which needs `python -m playwright install chromium`. The gallery is executed as the pages are built, and the fast-spin-echo
scripts need `blochsim` from the `design` extra.

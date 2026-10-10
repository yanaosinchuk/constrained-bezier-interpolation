# Technical report

This directory contains the LaTeX source and compiled PDF for the accompanying
technical report.

## Files

- `main.tex` — main LaTeX document
- `references.bib` — bibliography
- `constrained_bezier_interpolation.pdf` — compiled report
- `figures/` — figures and screenshots used by the report

The numerical comparison figures

- `figures/Compare.png`
- `figures/Residual_comparison.png`

are generated from the current Python implementation.

## Rebuild the numerical figures

From the repository root:

```bash
python examples/compare_interpolation_methods.py \
    --save-plots paper/figures \
    --no-show
```

## Compile the report

A TeX installation with pdfLaTeX, Biber, and `latexmk` is recommended.

From this directory:

```bash
latexmk -pdf main.tex
```

The equivalent manual compilation sequence is:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The committed PDF and numerical comparison figures correspond to the current
implementation and corrected cubic baseline.

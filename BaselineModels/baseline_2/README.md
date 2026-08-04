# Stage 1 IEEE baseline report

This project is a complete IEEE conference-style LaTeX progress report for the implemented Stage 1 instrument concept-learning baseline.

## Compile

From this directory, run:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The project requires a standard TeX distribution with the `IEEEtran`, `tikz`, and `pgfplots` packages. `training_history_40_epochs.csv` is included because the figures load it directly through pgfplots.

Replace the author placeholder in `main.tex` before submission. Dataset-specific references can be added to `references.bib` once their bibliographic details are available.

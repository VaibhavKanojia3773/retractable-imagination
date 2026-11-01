# LaTeX paper

**Sole author: Vaibhav Kanojia.** Main source: `main.tex`. Built PDF: `main.pdf`.

The official `neurips_2026.sty` was copied byte-for-byte from the author's existing NeurIPS 2026 template kit. It has not been modified. SHA-256:

`c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`

The working paper uses `sglblindworkshop` with the ESR workshop title and no `final` option. Confirm the organizer's actual anonymity and page-limit requirements before submission; change to `dblblindworkshop` if an anonymous review copy is required. No acceptance is implied by the style selection.

## Build with local TeX

On Windows, run `build.cmd`. It uses the same pdflatex -> BibTeX -> pdflatex -> pdflatex sequence as the author's existing MiKTeX setup, without requiring Perl or latexmk.

On any platform:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Regenerate numbers and vector plots

Run `python scripts/generate_assets.py`, then rebuild. This copies the measured study JSON into `data/`, generates `numbers.tex`, produces vector PDFs under `figures/`, and writes result tables under `tables/`. It does not rerun or alter the experiments. NumPy and Matplotlib are required only for asset regeneration.

The architecture is editable TikZ in `figures/architecture.tex`. References are in `references.bib`. Data snapshots are included to make the tables auditable. The claims distinguish measured systems behavior from the still-open question of learned non-additive state revision.

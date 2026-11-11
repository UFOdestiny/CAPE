# Benchmark tables and figures

This Python package renders tables and figures from `$RESULTS_DIR` and saved scores under `$OUTPUT_DIR`.
It writes directly into the configured LaTeX project: `$PAPER_DIR/tables/` and `$PAPER_DIR/figs/`, defaulting to
`latex/`. PNG previews go to `$PREVIEW_DIR` (`artifacts/previews`); compilation intermediates go to `$BUILD_DIR`.

```bash
python -m visualization.build
python -m visualization.build --tables
python -m visualization.build --figures
python -m visualization.build --only summary teaser
# The separate paper project can then be compiled:
make -C latex
```

Requires the packages in `requirements.txt`, `latexmk`, XeLaTeX and a TeX installation with `type1cm` and Times fonts.
The renderer reads existing benchmark artifacts; some shared statistics require CPU computation. It never starts
GPU inference or an API request. Missing study results may prevent the corresponding table from being built.

| Path | Responsibility |
|---|---|
| `data.py` | Shared presentation statistics, significance tests and worked example |
| `tables/<section>.py` | One function per table, registered in that module's `TABLES` |
| `tables/layout.py` | Text-width panel layout and paired AUROC/Brier cells |
| `tables/fmt.py` | Cell formatting and printed method names |
| `figs/main.py`, `figs/appendix.py` | One function per figure, registered in `FIGURES` |
| `figs/framework/framework.tex` | Editable TikZ source for the method diagram |
| `figs/framework/framework.py` | Compile TikZ using the configured build directory |
| `style.py` | Widths, fonts, palette, PDF export checks and previews |
| `build.py` | Table/figure selection and output dispatch |

Table labels derive from function names: `summary_full()` writes `summary-full-panel-1.tex`.
Multi-panel figures write a PDF per label and a separate legend where needed. `panels.tex` records printed widths.
Generated PDFs, PNG previews and TeX intermediates go to the output directories above.

Figures are drawn at printed size: column 7.7 cm, text width 16 cm. Text embeds Times New Roman at the
ACL body's 10.95 TeX pt (10.909 PDF pt), with export checks for exact label size and text color. Math indices
retain their normal size ratios. Panels carry no internal titles. Baseline
colors use the shared blue, teal, orange and purple palette. Axes and inward ticks are black, and plots omit background grids.
Tables mark the best in bold and the second in underline, use native small
text, and preserve signs and confidence intervals.

The per-setting AUROC/Brier tables mark CAPE's significant differences from the best of the 17 main-comparison
estimators for each setting and metric (two-sided paired question bootstrap, 5% Benjamini–Hochberg FDR).
Building these tables also writes `results/paired_significance.csv`, with the rival identities, gains (positive
favors CAPE), 95% intervals, p-values and significance flags. Pooled Math rows are outside the testing families.

Run from the repository root. Set `PAPER_DIR`, `PREVIEW_DIR` and `BUILD_DIR` to temporary locations for isolated
rendering checks (for example, to diff rebuilt tables against the paper's). See [configuration](../docs/configuration.md) and [architecture](../docs/architecture.md).

# baseline_fast implementation report

`baseline_fast_explained.pdf` explains the two execution optimizations, their
relationship to the bucket degradation mechanisms, the saved full Replica
measurements, and the limits of the engineering/research claims.

The report does not run new experiments or change the SLAM implementation.
Tables and the runtime plot are regenerated from saved results. Input and tested
source hashes are recorded in `evidence_manifest.json`.

To regenerate from the repository root:

```bash
MPLCONFIGDIR=/tmp/coslam-report-matplotlib python3 docs/baseline_fast_explained/build_report.py
cd docs/baseline_fast_explained
pdflatex -interaction=nonstopmode -halt-on-error baseline_fast_explained.tex
pdflatex -interaction=nonstopmode -halt-on-error baseline_fast_explained.tex
```

The generator requires Python, NumPy and Matplotlib. PDF compilation requires
LaTeX with the packages listed in the source. The PDF includes clickable links
to the primary CUDA/PyTorch documentation and the example systems research call.

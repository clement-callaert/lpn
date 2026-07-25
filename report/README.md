# Research report: RL for test-time compute in LPN

## Purpose

Compact research report prepared for discussion with Nathanaël Fijalkow.
It documents a controlled stop-or-continue reinforcement-learning prototype for
allocating latent-search compute in frozen Latent Program Networks.

## Repository

- Fork: `https://github.com/clement-callaert/lpn`
- Branch: `research/stochastic-latent-search`
- Upstream: `https://github.com/clement-bonnet/lpn`

## Source artifacts

Primary matched evaluation (local; may be gitignored):

```text
artifacts/rl/results/settingB_ckpt2_ds0_len24_eval_20260723.json
```

Policy training summary used for hyperparameters:

```text
artifacts/rl/policies/settingB_penalty0.00_20260723/train_summary.json
```

## Figure provenance

| Report asset | Source |
| --- | --- |
| `data/controller_tradeoff.csv` | Regenerated from the Setting B JSON full controller set (Table 3) |
| `figures/controller_tradeoff.png` | Plotted from `data/controller_tradeoff.csv` |
| `data/stopping_distribution.csv` | Regenerated from Setting B `stop_histogram` fields |
| `figures/stopping_distribution.png` | Plotted from `data/stopping_distribution.csv` |

Committed presentation assets under `artifacts/readme/` are related but not the
report figure sources after regeneration.

## Compile

`latexmk` is not required. From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is available:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Final PDF

```text
report/main.pdf
```

## Page limit

The main text (title through Conclusion) must not exceed six pages.
References are excluded from the six-page limit.

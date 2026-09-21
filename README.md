# Scale-dependent ranking of relative inequality across binary health indicators

Analysis code for the manuscript *Scale-dependent ranking of relative inequality across
binary health indicators*, a secondary analysis of the seventh Korea National Health and
Nutrition Examination Survey (KNHANES VII, 2016-2018).

The scripts reproduce every estimate reported in the manuscript: Tables 1-4, Figure 1 and
Supplementary Tables S1-S10.

## Data

The survey microdata are **not** redistributed here. They are publicly available from the
Korea Disease Control and Prevention Agency to registered users at
<https://knhanes.kdca.go.kr>.

1. Download the integrated oral examination file of the seventh cycle, `HNY7_OE_16_18.sav`.
2. Place it, unmodified, in `data/raw/`.

No variable names or values in the source file are altered outside the scripts.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python 01_import_and_sample.py      # read the .sav file, define the analytic samples
python 02_construct_outcomes.py     # 11 indicators, 5 threshold variants, favourable recoding
python 03_construct_sep_rank.py     # weighted fractional (ridit) socioeconomic rank
python 04_primary_analysis.py       # three estimands, slope index, decomposition
python 05_bootstrap.py              # 2000 design-based bootstrap replicates, rank agreement
python 06_sensitivity_analyses.py   # seven sets of sensitivity analyses
python 07_tables.py                 # Tables 1-4 and Supplementary Tables S1-S10 (CSV)
python 08_figure1.py                # Figure 1 (PNG, 400 dpi)
```

Intermediate objects are written to `output/` and the final tables and figure to
`output/tables/` and `output/figures/`.

Running scripts 01 to 08 in order takes about 25 minutes on a laptop; script 05 dominates
the run time because the whole estimation sequence, including the rank transformation and
the ranking of indicators, is repeated in each of the 2000 replicates.

## Reproducibility notes

* Bootstrap seeds are fixed in `config.py`, so all reported estimates are exactly
  reproducible.
* Weighted logistic models are fitted by Newton-Raphson iteration in `inequality.py`; the
  point estimates were verified against the binomial generalised linear model routine of
  statsmodels with frequency weights, agreeing to four decimal places
  (`tests/test_against_statsmodels.py`).
* Tables 1, 3 and 4 and the decomposition and recoding analyses are generated from a single
  set of replicates, so confidence intervals for the same indicator are identical across
  tables.

## Repository contents

| File | Purpose |
|---|---|
| `config.py` | paths, outcome lists, covariate sets, bootstrap seeds |
| `inequality.py` | estimation engine: weighted logistic fit, ridit rank, estimands, agreement statistics |
| `01`-`08` | analysis pipeline, run in order |
| `tests/` | verification of the model fit against statsmodels |

## Citation

If you use this code, please cite the archived release (see `CITATION.cff`).

## Licence

MIT (see `LICENSE`).

# Changelog

All changes to the analysis code relative to the archived release are recorded here.

## [1.1.0] - 2026-09-21

Version accompanying the revised manuscript. Changes relative to v1.0.0:

### Changed
- `05_bootstrap.py`: variance estimation switched from a with-replacement bootstrap that
  drew the observed number of primary sampling units per stratum (weights unchanged) to the
  Rao-Wu rescaling bootstrap, which draws n - 1 units per stratum and multiplies the weights
  of the selected units by n / (n - 1). Point estimates are unchanged; confidence intervals
  are slightly wider.
- `inequality.py`: the stored replicate measures now include the two standardised
  prevalences (`p_advantaged`, `p_disadvantaged`).
- `07_tables.py`: favourable-coding prevalence ratios are now computed within the same
  bootstrap replicates as the adverse-coding estimates, from the two standardised
  prevalences, so that the adverse-versus-favourable comparison has a confidence interval.

### Changed (round 3 of internal review)
- `08_figure1.py`: Figure 1 is now a three-column slope chart (conditional odds ratio,
  marginal odds ratio, prevalence ratio) so that the non-collapsibility step and the scale
  step are shown separately, with rank numbers printed at the left of each panel; the gap
  between panels is narrower and the figure is written as PNG and TIFF at 600 dpi.
- `config.py`: the age-stratified analysis now includes ages 19-34 years (six age groups).
- `README.md`, `CITATION.cff`: title updated to match the manuscript.

### Added
- `07_tables.py`: generation of Supplementary Tables S2 (complete-case sample), S3
  (analytic reversal grid on the marginal odds ratio), S7 (covariate sets), S8 (socioeconomic
  rank definitions), S9 (monotonicity and quadratic rank) and S10 (predicted-probability
  indices), so that every supplementary table is now produced by the code.
- `inequality.py`: `favourable_prevalence_ratio()` and `taylor_ci()`, a Taylor-linearised
  95% confidence interval for the conditional odds ratio using the stratified cluster
  sandwich estimator.
- `06_sensitivity_analyses.py`: Taylor-linearised intervals for the 11 indicators on the
  income axis, as a check on the bootstrap.
- `07_tables.py`: Supplementary Table S11 (Taylor-linearised versus bootstrap intervals).
- `07_tables.py`: Supplementary Table S6 now reports, within each age group, the marginal odds
  ratio and the non-collapsibility and scale components alongside the conditional odds ratio
  and the prevalence ratio.
- `README.md`: description of the Rao-Wu bootstrap and the Taylor check; DOI badge and
  citation of the archived release.

## [1.0.0] - 2026-09-21
- Initial release accompanying the submitted manuscript.
  DOI: https://doi.org/10.5281/zenodo.22866369

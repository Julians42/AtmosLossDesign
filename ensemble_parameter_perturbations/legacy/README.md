# legacy/

Superseded regression-coefficient / SVD-based analysis of the same
simulation output that `pipeline/` analyzes with the information-gain
approach. This directory is kept for reference (some plots/derivations here
predate the information-gain method and may still be useful to compare
against), but it is **not actively maintained** - paths and includes were
fixed up during the 2026-08 reorg so the scripts parse and their obvious
bugs are fixed, but the analysis itself has not been re-validated.

## Contents

- `regression_funcs.jl` - the regression/SVD-specific functions that used to
  live in the top-level `helper_funcs.jl` (`plot_regression_analysis`,
  `plot_variable_informing_analysis`, `compute_regression_coefficients_optimized`,
  `create_parameter_dataframe`). Reuses `src/methods.jl` / `src/parameter_io.jl`
  for the pieces that are still shared with the active pipeline.
- `elbow_calculation.jl` - dimensionality/elbow-detection helpers
  (`dimension_by_kneedle`, `elbow_second_derivative`, `elbow_percentage_cutoff`),
  used only by `evaluate.jl`.
- `evaluate.jl` - main regression-analysis driver. Reads a legacy
  concatenated CSV (`data_dataframes/diagnostic_edmfx.csv`) and produces
  scree/bar-chart diagnostics per variable-subset scenario.
- `postprocessing/run_postprocessing.jl` - serial predecessor of the active
  `pipeline/03_postprocess/multi_postproc.jl`; processes all members in one
  pass into a single concatenated CSV instead of one-CSV-per-member.
- `postprocessing/resolution_effect.jl`, `postprocessing/how_low_can_you_go.jl` -
  resolution-vs-information-content analysis via regression coefficients
  (compare against the active `pipeline/05_plots/resolution.jl`, which
  answers the same question via information gain). Both reference an
  `output_4_diagnostic_edmfx` directory that predates the current
  `output_5_cfsites` experiment and does not exist on disk - verify/update
  that path before running.
- `postprocessing/information_metrics.jl` - `InformationMetric` type
  hierarchy (trace / effective-rank / log-determinant) used by
  `resolution_effect.jl`.
- `data_dataframes/` - the legacy single-file concatenated CSVs
  (`diagnostic_edmfx.csv`, `resolution_effect.csv`), as opposed to the active
  pipeline's one-CSV-per-member layout in `../data/postprocessing/`.
- `old/` - older, fully-dead scratch work (notebooks, a prognostic-EDMF
  config, a prior TOML for a different parameterization) that predates even
  the regression-coefficient approach.
- `tomls/` - parameter TOML fragments not reachable from the active
  `config/experiment_config.yml` or `config/diagnostic_edmfx_diurnal_scm_imp.yml`.
- `prognostic_edmfx_diurnal_scm_imp.yml` - SCM config for the prognostic
  EDMF scheme (the active pipeline only uses the diagnostic scheme).

## Bugs fixed while archiving (2026-08 reorg)

These scripts had pre-existing bugs that would have made them error out if
run as committed; fixed while moving them here since the fix was
unambiguous from context:

- `evaluate.jl` called `compute_regression_coefficients` (only ever defined
  commented-out) instead of `compute_regression_coefficients_optimized`; same
  in `resolution_effect.jl` and `how_low_can_you_go.jl`.
- `evaluate.jl` called `get_all_variables(config)` with one argument, but
  `get_all_variables` only ever had 2-argument (config, trait) methods; fixed
  to `get_all_variables(config, Config_cfsites())`, matching the flat
  `z_levels` list these ad hoc configs use.
- `evaluate.jl`'s first `plot_regression_analysis(...)` call passed
  `config["exp_name"]` (a `String`) where the function expects the full
  `config::Dict`.
- `run_postprocessing.jl` called `postprocess_dataframe(df)` with one
  argument, but the only `postprocess_dataframe` in scope took 4 arguments;
  fixed to call the (now-shared) `normalize_statistics(df)`.
- `information_metrics.jl` ran its mock-data `main()` demo unconditionally at
  `include` time, printing irrelevant output every time `resolution_effect.jl`
  included it; now guarded by `abspath(PROGRAM_FILE) == @__FILE__`.

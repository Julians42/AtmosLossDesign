# ensemble_parameter_perturbations

Quantifies how much each observation (and each vertical resolution / variable
subset) constrains the diagnostic-EDMF SCM's calibratable parameters, using
an information-gain analysis over a 100-member x 119-site ensemble.

## Layout

```
config/         experiment_config.yml + model/prior/forcing TOML/YAML inputs
src/            reusable Julia code, split by what it depends on:
                  methods.jl       pure math/stats - no I/O, no schema assumptions
                  parameter_io.jl  file I/O given explicit paths
                  experiment.jl    glue: output_5_cfsites layout + experiment_config.yml schema
                  plotting_theme.jl shared CairoMakie theme (opt-in, plot scripts only)
pipeline/       driver scripts, one subfolder per pipeline stage (run in order, see below)
data/           ensemble output + derived intermediate data (all gitignored)
plots/          finished figures
legacy/         superseded regression-based analysis, kept for reference (see legacy/README.md)
logs/           SLURM stdout/stderr
```

Every script resolves its own paths relative to its own location
(`@__DIR__`) or a `PROJECT_ROOT` computed from it, so **you can run any
pipeline script from any working directory** - you no longer need to `cd`
into this directory first (though sbatch scripts still do, since that's also
where `--project=.` picks up `Project.toml`).

## Pipeline

Run these in order. Steps 1-2 are the expensive, cluster-only steps; if
`data/` already has output from a previous run, you can often start from
step 4 or 5.

1. **Configure** `config/experiment_config.yml` (variable names, z-levels,
   output directory, prior/forcing TOML paths).
2. **Run the SCM ensemble**: `sbatch pipeline/02_ensemble/submit_ensemble.sbatch`
   (edit `pipeline/02_ensemble/run_ensemble.jl` first if you need to change
   site selection or ensemble size). Produces
   `data/output_5_cfsites/member_NNN/location_.../output_active/*.nc` plus
   each member's `parameter.toml`. (`pipeline/01_forcing/` generates the
   ERA5-derived forcing files this step consumes, if they don't already
   exist in the artifact cache.)
3. **Postprocess** each member's output into per-member statistics CSVs, in
   parallel: `sbatch pipeline/03_postprocess/submit_postproc.sbatch` (a
   100-way SLURM array job running `multi_postproc.jl $SLURM_ARRAY_TASK_ID`).
   Produces `data/postprocessing/diagnostic_edmfx/member_NNN.csv`.
4. **Analysis**: compute the information-gain Jacobian/covariances.
   - `julia --project=. pipeline/04_analysis/point_estimate.jl` - the
     non-bootstrapped point estimate, over the full (non-resampled) dataset.
     Writes `data/bootstrap_sites/no_bootstrap/results.jld2`.
   - `julia --project=. pipeline/04_analysis/bootstrap_sites.jl` - resamples
     sites with replacement 100x to quantify uncertainty. Writes
     `data/bootstrap_sites/bootstrap_1..100/results.jld2`.
5. **Plot**, from `pipeline/05_plots/`:
   - `marginals.jl` - marginal information gain per observation and
     parameter learnability, with bootstrap uncertainty -> `plots/marginal_info_gain_full_resolution_with_uncertainty.png`.
   - `resolution.jl` - effect of vertical resolution on information content
     (information tapers off around 400-500m) -> `plots/resolution_information_gain.png`.
   - `greedy_loss_function.jl` - greedy forward-selection ordering of
     variables by information gain, with bootstrap uncertainty -> `plots/greedy_loss_function*.png`.
   - `experiment_scenarios.jl` - compares realistic observation scenarios
     (dropping cloud profiles, adding TKE/entrainment/updraft area) against
     the full high-resolution case -> `plots/experiment_parameter_informedness.png`.

   `marginals.jl` and `greedy_loss_function.jl` cache intermediate results in
   `data/cache/*.csv` (bootstrap loops are slow); delete the relevant cache
   file to force a full recompute rather than a resume/append.

All scripts are run with the `Project.toml` at this directory's root:
`julia --project=<path to this dir> <script>`.

## The two analysis generations

This directory has two analysis approaches layered on the same simulation
output:

- **Information-gain** (active, described above): `src/methods.jl` +
  `src/experiment.jl`, driven from `pipeline/04_analysis/` and
  `pipeline/05_plots/`. This is what's actively maintained and what the
  README above documents.
- **Regression-coefficient/SVD** (legacy): superseded by the above, moved to
  `legacy/` with its own README. Still runnable, but not actively maintained
  - see `legacy/README.md` before relying on it.

## Reusing the analysis code elsewhere (e.g. without `output_5_cfsites`)

If you already have parameters/statistics in memory (e.g. loaded from your
own CSV) and just want the information-gain math, use `src/methods.jl`
directly - it has no file I/O or directory-layout assumptions:

```julia
include("src/methods.jl")
ig, ∇G, Σ_y, Σ_0 = information_gain(variables, df, constrained_params, param_ordering, Linear())
```

`src/parameter_io.jl` has the path-based loading (`load_member_parameters`,
`load_raw_parameter_toml`) if you do want to read from a
`member_NNN/parameter.toml` layout but not the rest of the
`experiment_config.yml`-driven glue in `src/experiment.jl`.

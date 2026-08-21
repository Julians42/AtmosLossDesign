# Quantifying Observations' ability to constrain climate model parameters and climate statistics

## Workflow

The `ensemble_parameter_perturbations/` experiment (100-member x 119-site
diagnostic-EDMF SCM ensemble) is the actively maintained one. See
[`ensemble_parameter_perturbations/README.md`](ensemble_parameter_perturbations/README.md)
for the full pipeline: configure -> run the SCM ensemble -> postprocess ->
information-gain analysis -> plots, plus how to reuse the analysis code
against your own data.

`single_parameter_perturbations/` holds a separate, related experiment
(single/multi-parameter gradient studies); it currently has its own flat,
undocumented layout and depends on `ensemble_parameter_perturbations`'s
`data/output_5_cfsites` and `data/bootstrap_sites/no_bootstrap` outputs - not
yet reorganized.

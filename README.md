# Quantifying Observations' ability to constrain climate model parameters and climate statistics

## Workflow
1. Configure the experiment file `experiment_config.yml`. These are args that are used throughout the experiment.
2. Run the SCM setup using `ensemble_parameter_perturbations/submit_ensemble.sbatch`. You should edit the associated `ensemble_parameter_perturbations/run_ensemble.jl` file to configure the output directory / sites etc.
3. Process the observations into statistics. We do this in parallel since ~50k simulations is quite slow serially. The two scripts you need are `ensemble_parameter_perturbations/postprocessing/submit_postproc.sbatch` and `run_postprocessing.jl`.
Now you can make plots!
### Plotting Scripts 
1. Most plotting utilities are in `ensemble_parameter_perturbations/new_helper_funcs.jl`. Plotting scripts can be found in `ensemble_parameter_perturbations/plot_routines`.
2. `marginals.jl`: Plots the marginal information gain of each observation with uncertainty at specified resolution. We use a bootstrap to compute the uncertainty. The samples for the bootstrap are generated using `ensemble_parameter_perturbations/bootstrap_sites.jl`.
3. `resolution.jl`: Estimates the effect of resolution on the information content of observations. We find that roughly 400-500 meter resolution is where the information starts to taper off. 
4. `realistic/realistic_greedy_bs.jl`: Estimates the ordering of variable information content importance with bootstrap uncertainty.
5. `realistic/realistic_marginals.jl`: Plot of different experiments and the information content they provide to observations. 
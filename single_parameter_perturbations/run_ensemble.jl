using TOML 
using Distributed
using YAML
using NCDatasets

using Dates

import EnsembleKalmanProcesses as EKP
import ScikitLearn
import JLD2
import ClimaCalibrate as CAL
using Statistics
using Glob
using ClimaAnalysis

num_procs = 400
run_time = "10:00:00"

# include("model_interface.jl")

include("new_helper_funcs.jl")
experiment_config = YAML.load_file("experiment_config.yml")
quantiles = [0.25, 0.5, 0.75]
default_toml = TOML.parsefile("tomls/mean_prior_diagnostic_pi_entr_smooth_entr_detr.toml")
# run over multiple seasons
start_dates = ["20070101", "20070401", "20070701", "20071001"]

lats, lons, relaxation_tomls = get_all_cfsites()

quantile_matrix, param_names = param_perturbations(experiment_config["prior_path"])
toml_list = []
for param in param_names
    for q in quantiles
        # for each setup generate a perturbed toml file
        perturbed_toml = get_perturbed_toml(param, 
        q;
        default_toml = default_toml,
        q_mat = quantile_matrix,
        param_names = param_names,
        quantiles = quantiles)
        # save the perturbed toml file
        save_dir = joinpath(experiment_config["output_dir"], param, string(q))
        if !isdir(save_dir)
            mkpath(save_dir)
        end
        open(joinpath(save_dir, "parameters.toml"), "w") do io
            TOML.print(io, perturbed_toml)
        end
        push!(toml_list, joinpath(save_dir, "parameters.toml"))

    end
end
@info "Total number of toml files: $(length(toml_list))"

addprocs(
    CAL.SlurmManager(num_procs),
    t = run_time,
    mem_per_cpu = "25G",
    cpus_per_task = "1",
)


# Distribute required code and packages
@everywhere using TOML
@everywhere include("model_interface.jl")
@everywhere experiment_config = $(experiment_config)

# Send toml_list to each worker
for p in workers()
    @eval @spawnat $p global toml_list = $(toml_list)
end

println("Going to run $(length(toml_list) * length(lats) * length(start_dates)) simulations...")

default_worker_pool() = WorkerPool(workers())

run_iteration(toml_list, 
            length(toml_list), 
            lats, 
            lons, 
            start_dates; 
            worker_pool = default_worker_pool(),
            experiment_config = experiment_config
)

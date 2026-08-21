using TOML
using Distributed
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate as CAL
using Glob
using NCDatasets
import YAML

num_procs = 800
ensemble_size = 100

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
experiment_config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
output_dir = joinpath(PROJECT_ROOT, experiment_config["output_dir"])
toml_path_name = experiment_config["toml_path_name"]
prior = CAL.get_prior(joinpath(PROJECT_ROOT, experiment_config["prior_path"]))

#################### Site Selection ###########################
ds = NCDataset(joinpath(dirname(PROJECT_ROOT), "coszen_data.nc"))

deep_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)
shallow_sites = setdiff(collect(1:119), deep_sites)

lats, lons, relaxation_tomls = [], [], []
for site in 1:119
    push!(lats, ds["lat"][site])
    push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
    if site in deep_sites
        push!(relaxation_tomls, joinpath(PROJECT_ROOT, experiment_config["forcing_toml_files"]["deep"]))
    else
        push!(relaxation_tomls, joinpath(PROJECT_ROOT, experiment_config["forcing_toml_files"]["shallow"]))
    end
end

start_dates = ["20071001"]
#################### Site Selection ###########################

# sample from the parameter distribution and write the elements to the toml files.
# param_array = EKP.construct_initial_ensemble(prior, ensemble_size)
# EKP.TOMLInterface.save_parameter_ensemble(param_array, prior, CAL.get_param_dict(prior), output_dir, toml_path_name)
toml_list = glob(joinpath(output_dir, "member_*", toml_path_name * ".toml"))

# add processes
addprocs(
    CAL.SlurmManager(num_procs),
    t = "12:00:00",
    mem_per_cpu = "25G",
    cpus_per_task = "1",
)

# Distribute required code and packages. model_interface.jl is loaded via an
# absolute path computed on this (the launching) process, since workers may
# not share this process's working directory.
model_interface_path = joinpath(@__DIR__, "model_interface.jl")
@everywhere using TOML
@everywhere include($model_interface_path)

# Send toml_list to each worker
for p in workers()
    @eval @spawnat $p global toml_list = $(toml_list)
end

println("Going to run $(length(toml_list) * length(lats) * length(start_dates)) simulations...")

default_worker_pool() = WorkerPool(workers())

run_iteration(toml_list, length(toml_list), lats, lons, start_dates, relaxation_tomls)

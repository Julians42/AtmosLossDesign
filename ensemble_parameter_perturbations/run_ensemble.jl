using TOML 
using Distributed
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate as CAL
using Glob
using NCDatasets
import YAML

num_procs = 800
ensemble_size = 100

# experiment_dir = dirname(Base.active_project())
experiment_dir = "/central/groups/esm/jschmitt/experiments/AtmosLossDesign/ensemble_parameter_perturbations"
experiment_config = YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
output_dir = experiment_config["output_dir"]
toml_path_name = experiment_config["toml_path_name"]
prior = CAL.get_prior(joinpath(experiment_dir, experiment_config["prior_path"]))

#################### Site Selection ###########################
# lats = [
#     -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -18.5, -17.0,
#     -15.5, -14.0, -12.5, -11.0, -9.5, -8.0, 35.0, 32.0,
#     29.0, 23.0, 20.0, 17.0
# ]

# lons = [
#     -72.5, -75.0, -77.5, -80.0, -82.5, -85.0, -90.0, -95.0,
#     -100.0, -105.0, -110.0, -115.0, -120.0, -125.1000061,
#     -125.0, -129.0, -133.0, -141.0, -145.0, -149.0
# ]

ds = NCDataset("../coszen_data.nc")

deep_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)
shallow_sites = setdiff(collect(1:119), deep_sites)

lats, lons, relaxation_tomls = [], [], []
for site in 1:119
    push!(lats, ds["lat"][site])
    push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
    if site in deep_sites
        push!(relaxation_tomls, experiment_config["forcing_toml_files"]["deep"])
    else
        push!(relaxation_tomls, experiment_config["forcing_toml_files"]["shallow"])
    end
end

start_dates = ["20071001"] # , 
#################### Site Selection ###########################


# sample from the parameter distribution and write the elements to the toml files. 
# param_array = EKP.construct_initial_ensemble(prior, ensemble_size)
#EKP.TOMLInterface.save_parameter_ensemble(param_array, prior, CAL.get_param_dict(prior), output_dir, toml_path_name)
toml_list = glob(joinpath(output_dir, "member_*", toml_path_name * ".toml"))

# add processes 
addprocs(
    CAL.SlurmManager(num_procs),
    t = "12:00:00",
    mem_per_cpu = "25G",
    cpus_per_task = "1",
)


# Distribute required code and packages
@everywhere using TOML
@everywhere include("model_interface.jl")

# Send toml_list to each worker
for p in workers()
    @eval @spawnat $p global toml_list = $(toml_list)
end

println("Going to run $(length(toml_list) * length(lats) * length(start_dates)) simulations...")

default_worker_pool() = WorkerPool(workers())

run_iteration(toml_list, length(toml_list), lats, lons, start_dates, relaxation_tomls)

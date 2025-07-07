using TOML 
using Distributed
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate as CAL
using Glob

num_procs = 400
ensemble_size = 100

lats = [
    -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -18.5, -17.0,
    -15.5, -14.0, -12.5, -11.0, -9.5, -8.0, 35.0, 32.0,
    29.0, 23.0, 20.0, 17.0
]

lons = [
    -72.5, -75.0, -77.5, -80.0, -82.5, -85.0, -90.0, -95.0,
    -100.0, -105.0, -110.0, -115.0, -120.0, -125.1000061,
    -125.0, -129.0, -133.0, -141.0, -145.0, -149.0
]
start_dates = ["20070101", "20070401", "20070701", "20071001"]

output_dir = "output_4_diagnostic_edmfx"
toml_path_name = "parameter"
prior = CAL.get_prior("priors/prior_diagnostic_pi_entr_smooth_entr_detr_coarse_amip_new.toml")

# sample from the parameter distribution and write the elements to the toml files. 
param_array = EKP.construct_initial_ensemble(prior, ensemble_size)
EKP.TOMLInterface.save_parameter_ensemble(param_array, prior, CAL.get_param_dict(prior), output_dir, toml_path_name)

# get the toml_list 

toml_list = glob(joinpath(output_dir, "member_*", toml_path_name * ".toml"))

addprocs(
    CAL.SlurmManager(num_procs),
    t = "12:00:00",
    mem_per_cpu = "12G",
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

run_iteration(toml_list, length(toml_list), lats, lons, start_dates)

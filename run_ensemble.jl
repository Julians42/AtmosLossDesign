using TOML 
using Distributed

include("model_interface.jl")

perturbations = collect(.5:.05:1.5)#[0.8, 1.0, 1.2]
default_toml = TOML.parsefile("tomls/simple_precalibrated.toml")

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

toml_list = []
exp_root = "ss_perturbed"
if !isdir(exp_root)
    mkdir(exp_root)
end
# default_toml[collect(keys(default_toml))[1]]["value"]
for (k, v) in default_toml
    println(k)
    if !isdir(joinpath(exp_root, k))
        mkdir(joinpath(exp_root, k))
    end
    #println(v["value"])
    for perturbation in perturbations
        cptoml = deepcopy(default_toml)
        println("perturbation: ", perturbation)
        cptoml[k]["value"] = v["value"] .* perturbation
        println(cptoml[k]["value"])
        if !isdir(joinpath(exp_root, k, string(perturbation)))
            mkdir(joinpath(exp_root, k, string(perturbation)))
        end
        # save new cptoml to this directory
        open(joinpath(exp_root, k, string(perturbation), "parameters.toml"), "w") do io
            TOML.print(io, cptoml)
        end
        push!(toml_list, joinpath(exp_root, k, string(perturbation), "parameters.toml"))
    end
    println(" ")
end

addprocs(
    CAL.SlurmManager(80),
    t = "2:00:00",
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

println("Going to run $(length(toml_list)) simulations...")

default_worker_pool() = WorkerPool(workers())

run_iteration(toml_list, length(toml_list))

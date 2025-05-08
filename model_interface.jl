import ClimaAtmos as CA
import YAML
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: path_to_ensemble_member

import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
using JLD2

#include("helper_funcs.jl")

using Distributed
# const experiment_config_dict =
#     YAML.load_file(joinpath(@__DIR__, "experiment_config.yml"))
# const output_dir = experiment_config_dict["output_dir"]
# const model_config = experiment_config_dict["model_config"]
# const batch_size = experiment_config_dict["batch_size"]

@everywhere function run_atmos_simulation(atmos_config)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        if !isnothing(sol_res.sol)
            T = eltype(sol_res.sol)
            if T !== Any && isconcretetype(T)
                sol_res.sol .= T(NaN)
            else
                fill!(sol_res.sol, NaN)
            end
        end
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end

function forward_model(parameter_path)
    base_config_dict = YAML.load_file(joinpath(@__DIR__, "prognostic_edmfx_tv_era5driven_column.yml"))

    member_path = dirname(parameter_path)

    config_dict = deepcopy(base_config_dict)
    config_dict["output_dir"] = member_path
    if haskey(config_dict, "toml")
        config_dict["toml"] = abspath.(config_dict["toml"])
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end
    config_dict["output_default_diagnostics"] = false # ensure default diagnostics are off

    comms_ctx = ClimaComms.SingletonCommsContext()
    config = CA.AtmosConfig(config_dict; comms_ctx)

    # @info "Preparing to run $(length(atmos_configs)) model simulations in parallel."
    # println("Number of workers: ", nprocs())

    start_time = time()
    #map(run_atmos_simulation, atmos_configs)
    try
        run_atmos_simulation(config)
    catch e
        @warn "Simulation crashed for parameter file $(parameter_path): $(e)"
        return
    end
    end_time = time()

    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished all model simulations. Total time taken: $(elapsed_time) minutes."

end


function run_iteration(
    parameter_paths,
    ensemble_size; 
    worker_pool = default_worker_pool(),
)
    @sync begin 
        for m in 1:ensemble_size
            @async begin
                worker = take!(worker_pool)
                @show worker
                try
                    remotecall_wait(forward_model, worker, parameter_paths[m])
                catch e
                    @warn "Error in worker $(worker): $(e)"
                finally
                    put!(worker_pool, worker)
                end
            end
        end
    end
end

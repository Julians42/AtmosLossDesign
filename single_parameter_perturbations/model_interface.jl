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

function forward_model(parameter_path, lat, lon, start_date)
    base_config_dict = YAML.load_file(joinpath(@__DIR__, "prognostic_edmfx_tv_era5driven_column.yml"))
    config_dict = deepcopy(base_config_dict)

    # update the config_dict with site latitude / longitude
    config_dict["site_latitude"] = lat
    config_dict["site_longitude"] = lon
    config_dict["start_date"] = start_date

    # set the data output directory
    member_path = dirname(parameter_path)
    member_path = joinpath(member_path, "$(lat)_$(lon)_$(start_date)")
    config_dict["output_dir"] = member_path

    # add the perturbation toml to the config_dict
    if haskey(config_dict, "toml")
        config_dict["toml"] = abspath.(config_dict["toml"])
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    comms_ctx = ClimaComms.SingletonCommsContext()
    config = CA.AtmosConfig(config_dict; comms_ctx)

    start_time = time()
    try
        run_atmos_simulation(config)
    catch e
        @warn "Simulation crashed for parameter file $(parameter_path): $(e)"
        return
    end
    end_time = time()

    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished simulation. Total time taken: $(elapsed_time) minutes."

end


function run_iteration(
    parameter_paths,
    ensemble_size,
    lats, 
    lons,
    start_dates; 
    worker_pool = default_worker_pool(),
)
    @sync begin 
        for start_date in start_dates
            for (site_index, (lat, lon)) in enumerate(zip(lats, lons))
                for m in 1:ensemble_size
                    @async begin
                        worker = take!(worker_pool)
                        try
                            @show worker site_index m
                            # Pass lat and lon into forward_model
                            remotecall_wait(forward_model, worker, parameter_paths[m], lat, lon, start_date)
                        catch e
                            @warn "Error in worker $(worker) at site $(site_index) for start date $(start_date): $(e)"
                        finally
                            put!(worker_pool, worker)
                        end
                    end
                end
            end
        end
    end
end

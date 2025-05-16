using Dates
import ClimaAnalysis as AN

import EnsembleKalmanProcesses as EKP
import ScikitLearn
import YAML
import JLD2
import ClimaCalibrate as CAL
import CalibrateEmulateSample as CES
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis

config = YAML.load_file("experiment_config.yml")

parameters = glob("perturbed/*")
param_names = basename.(parameters)


function generate_perturbation_dict(short_name, perturb_amounts; zlev = nothing)
    profiles = Dict{String, Dict{String, Any}}()

    for (i, parameter) in enumerate(parameters)
        param_name = basename(parameter)
        profiles[param_name] = Dict()
        for perturb in perturb_amounts
            try
                # Read the parameter file
                sim_dir = AN.SimDir(joinpath(parameter, perturb, "output_active"))
                data = AN.get(sim_dir; short_name = short_name, reduction = "inst")
                if data.dims["time"][end] < config["reduction_end_time"]
                    profiles[param_name][perturb] = [NaN]
                else
                    profile = AN.window(data, "time", left = config["reduction_start_time"], right = config["reduction_end_time"])
                    averaged_profile = AN.average_time(slice(profile, x = 0, y = 0))

                    if !haskey(averaged_profile.dims, "z")
                        profiles[param_name][perturb] = averaged_profile.data
                    else
                        profiles[param_name][perturb] = AN.slice(averaged_profile, z = zlev).data
                    end
                    # Store the averaged profile in the nested dictionary
                end
            catch e 
                println("Error processing parameter: $param_name, perturbation: $perturb")
                println(e)
                profiles[param_name][perturb] = [NaN]
            end
        end
    end
    return profiles
end


function statistic_parameter_plot(statistic, zlev = 0; purturbations = string.(collect(.5:0.05:1.5)))
    # access the data
    profiles = generate_perturbation_dict(statistic, purturbations, zlev = zlev);

    fig = Figure(size = (250 * min(length(param_names), 5), 250 * ceil(Int, length(param_names) / 5)))  # Adjust figure size
    for (i, param_name) in enumerate(param_names)
        row = div(i - 1, 5) + 1  # Calculate row index
        col = mod(i - 1, 5) + 1  # Calculate column index
        ax = Axis(fig[row, col], title = param_name, xlabel = "Fraction of Prior Mean Value", ylabel = "$statistic at $zlev m")
        resultant_perturbed_amounts = collect(keys(profiles[param_name]))
        obs = [profiles[param_name][perturb][1] for perturb in resultant_perturbed_amounts]
        obs = hcat(obs...)[:]
        @info size(resultant_perturbed_amounts)
        valid_idx = .!isnan.(obs)
        xvals = parse.(Float64, resultant_perturbed_amounts)[valid_idx]
        obs = obs[valid_idx]
        scatter!(ax, xvals, obs, label = "Observations")
    end

    fig
    # save fig 
    save("figures/single_var_perturbations/$(statistic)_$(zlev).png", fig)
    return profiles
end





function nc_fetch(filename::String, var::String, obs_start::DateTime, obs_end::DateTime)
    NCDataset(filename, "r") do ds
        time = ds["time"][:]
        time_idx = findall(x -> x >= obs_start && x <= obs_end, time)
        return mean(ds[var][1, 1, :, time_idx], dims = 2)[:]
    end

end



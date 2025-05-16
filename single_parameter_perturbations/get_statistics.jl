import YAML
import JLD2
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis
using DataFrames
using CSV

# include("helper_funcs.jl")
config = YAML.load_file("experiment_config.yml")


all_vars = vcat([config["var_names_int"], config["var_names_prof"]]...)


# Define list to collect rows
rows = []

# get all possible combinations
parameters = basename.(glob("multiseason_smoothed/*"))
perturbations = collect(Set(basename.(glob("multiseason_smoothed/*/*"))))
sites = collect(Set(basename.(glob("multiseason_smoothed/*/*/*01"))))
all_paths = joinpath.(vec(collect(Iterators.product([parameters, perturbations, sites]...))))


for path in all_paths
    param_name, perturb, site = String.(split(path, "/"))
    perturb = parse(Float64, perturb)
    @info param_name, perturb, site

    sim_dir = SimDir(joinpath("multiseason_smoothed", path, "output_active"))
    try
        for data_var in all_vars
            data = get(sim_dir; short_name = data_var, reduction = "inst")

            if data.dims["time"][end] < config["reduction_end_time"]
                throw()
                profile = NaN  # or `missing`
            else
                profile_data = window(data, "time";
                                    left=config["reduction_start_time"],
                                    right=config["reduction_end_time"])
                averaged_profile = average_time(slice(profile_data, x=0, y=0))

                if !haskey(averaged_profile.dims, "z")
                    stat = averaged_profile.data[1]
                    push!(rows, (
                        param = param_name,
                        perturb = perturb,
                        site = site,
                        variable = data_var,
                        statistic = stat,
                    ))
                else
                    for zlev in config["z_levels"]
                        stat = slice(averaged_profile, z=zlev).data[1]
                        push!(rows, (
                            param = param_name,
                            perturb = perturb,
                            site = site,
                            variable = join([data_var, zlev], "_"),
                            statistic = stat,
                        ))
                    end
                end
            end
        end
    catch
        @info "Simulation faled for $param_name, $perturb, and $site. Appending NaNs..."
        push!(rows, (
            param = param_name,
            perturb = perturb,
            site = site, 
            variable = NaN,
            statistic = NaN,
        ))
    end
end

# Convert to DataFrame
df = DataFrame(rows)
CSV.write("multiseason_smoothed.csv", df)


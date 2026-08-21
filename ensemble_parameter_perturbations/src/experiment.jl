# Dataset-specific glue.
#
# Everything here assumes the `output_5_cfsites` on-disk layout, the
# experiment_config.yml schema, and the Config_cfsites* variable-naming
# convention. If you already have data loaded some other way, use
# methods.jl and parameter_io.jl directly instead - that's the reusable
# part.
#
# Include order matters: this file calls functions from methods.jl and
# parameter_io.jl, so include those first.
#
# Expects the including script to have already done:
#   using YAML, DataFrames, ClimaAnalysis
#   import EnsembleKalmanProcesses as EKP
#   import ClimaCalibrate as CAL

# Resolved lexically to this file's own location, so these defaults are
# correct no matter which script includes this file or what its CWD is.
_default_output_dir() = normpath(joinpath(@__DIR__, "..", "data", "output_5_cfsites"))
_default_prior_path() = normpath(joinpath(@__DIR__, "..", "config", "priors", "prior_diagnostic_pi_entr_smooth_entr_detr_coarse_amip_new.toml"))

"""
    resolve_config_paths!(config, project_root)

Rewrites the `output_dir`, `prior_path`, and `forcing_toml_files` entries of
a loaded experiment_config.yml dict to be absolute, resolved against
`project_root` (already-absolute entries are left untouched). Call this
immediately after loading the config so downstream code can use
`config["output_dir"]` etc. directly, regardless of the process's current
working directory.
"""
function resolve_config_paths!(config::Dict, project_root::String)
    if haskey(config, "output_dir") && !isabspath(config["output_dir"])
        config["output_dir"] = joinpath(project_root, config["output_dir"])
    end
    if haskey(config, "prior_path") && !isabspath(config["prior_path"])
        config["prior_path"] = joinpath(project_root, config["prior_path"])
    end
    if haskey(config, "forcing_toml_files")
        for key in keys(config["forcing_toml_files"])
            if !isabspath(config["forcing_toml_files"][key])
                config["forcing_toml_files"][key] = joinpath(project_root, config["forcing_toml_files"][key])
            end
        end
    end
    return config
end

abstract type AbstractConfig end
struct Config_cfsites <: AbstractConfig end
struct Config_cfsites_deep <: AbstractConfig end
struct Config_cfsites_shallow <: AbstractConfig end

# `z_levels` is a flat list, e.g. the ad hoc scenario configs in legacy/evaluate.jl
function get_all_variables(config::Dict, ::Config_cfsites)
    all_vars = String[]
    for var in config["var_names_prof"]
        for zlev in config["z_levels"]
            push!(all_vars, join([var, zlev], "_"))
        end
    end
    all_vars = vcat(all_vars, config["var_names_int"])
    return all_vars
end

# `z_levels` is a Dict("deep" => [start, step, stop], "shallow" => [...])
function get_all_variables(config::Dict, ::Config_cfsites_deep)
    all_vars = String[]
    for var in config["var_names_prof"]
        start, step, stop = config["z_levels"]["deep"]
        for zlev in collect(start:step:stop)
            push!(all_vars, join([var, zlev], "_"))
        end
    end
    all_vars = vcat(all_vars, config["var_names_int"])
    return all_vars
end

function get_all_variables(config::Dict, ::Config_cfsites_shallow)
    all_vars = String[]
    for var in config["var_names_prof"]
        start, step, stop = config["z_levels"]["shallow"]
        for zlev in collect(start:step:stop)
            push!(all_vars, join([var, zlev], "_"))
        end
    end
    all_vars = vcat(all_vars, config["var_names_int"])
    return all_vars
end

"""
    constrained_and_normalized_parameters(; n_members=100, rootdir, prior_path)

Loads the ensemble of parameters from `rootdir/member_NNN/parameter.toml` and
transforms them to standard-gaussian, normalized space. Thin composition of
`load_member_parameters` (parameter_io.jl) + `normalize_constrained_parameters`
(methods.jl) - use those directly if you already have parameters in memory.
"""
function constrained_and_normalized_parameters(; n_members::Int = 100,
        rootdir::String = _default_output_dir(),
        prior_path::String = _default_prior_path())
    prior = load_prior(prior_path)
    param_dicts = load_member_parameters(rootdir, n_members)
    return normalize_constrained_parameters(param_dicts, prior)
end

"""
    compute_parameter_statistics(prior_path::String; n_samples=10_000)

Path-taking convenience wrapper around the pure `compute_parameter_statistics(prior; ...)`
in methods.jl.
"""
compute_parameter_statistics(prior_path::String; n_samples::Int = 10_000) =
    compute_parameter_statistics(load_prior(prior_path); n_samples = n_samples)

"""
    information_gain(variables, df, grad_method; rootdir, prior_path)

Convenience wrapper matching the original call signature used throughout the
pipeline: loads and normalizes the parameter ensemble from `rootdir` /
`prior_path`, then calls the pure `information_gain` core in methods.jl.
Returns `(ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering)`.
"""
function information_gain(variables, df, grad_method::GradientApproximationMethod;
        rootdir::String = _default_output_dir(),
        prior_path::String = _default_prior_path())
    constrained_params, param_ordering = constrained_and_normalized_parameters(rootdir = rootdir, prior_path = prior_path)
    ig, ∇G, Σ_y, Σ_0 = information_gain(variables, df, constrained_params, param_ordering, grad_method)
    return ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering
end

"""
    process_members_sites(members, sites, param_stats_df, config)

Reads simulation output for each (member, site) pair under
`config["output_dir"]`, averages over the configured reduction time window,
and returns a long-format DataFrame of (member, site, variable, statistic,
convection_type). This is the per-member postprocessing step run by
pipeline/03_postprocess/multi_postproc.jl.
"""
function process_members_sites(members, sites, param_stats_df, config)
    rows = []

    data_vars = vcat(config["var_names_int"], config["var_names_prof"])

    for member in members
        @info "Processing member $member"
        norm_params = normalize_parameters("$(config["output_dir"])/$member/parameter.toml", param_stats_df)

        for site in sites
            site_data_path = joinpath(config["output_dir"], member, site, "output_active")
            sim_dir = SimDir(site_data_path)
            forcing_type = YAML.load_file(joinpath(site_data_path, ".yml"))

            toml_files = forcing_type["toml"]
            toml_string = join(toml_files, " ")
            if occursin("deep", toml_string)
                start, step, stop = config["z_levels"]["deep"]
                z_levels = collect(start:step:stop)
                convection_type = "deep"
            elseif occursin("shallow", toml_string)
                start, step, stop = config["z_levels"]["shallow"]
                z_levels = collect(start:step:stop)
                convection_type = "shallow"
            else
                @error "No deep or shallow forcing found in $site_data_path. Revise script..."
                convection_type = "unknown"
            end
            try
                for data_var in data_vars
                    data = get(sim_dir; short_name = data_var, reduction = "inst")

                    if data.dims["time"][end] < config["reduction_end_time"]
                        throw(ErrorException("Simulation time too short"))
                    else
                        profile_data = window(data, "time";
                                            left=config["reduction_start_time"],
                                            right=config["reduction_end_time"])
                        averaged_profile = average_time(slice(profile_data, x=0, y=0))

                        if !haskey(averaged_profile.dims, "z")
                            stat = averaged_profile.data[1]
                            push!(rows, (
                                member = member,
                                site = site,
                                variable = data_var,
                                statistic = stat,
                                convection_type = convection_type,
                            ))
                        else
                            for zlev in z_levels
                                stat = slice(averaged_profile, z=zlev).data[1]
                                push!(rows, (
                                    member = member,
                                    site = site,
                                    variable = join([data_var, zlev], "_"),
                                    statistic = stat,
                                    convection_type = convection_type,
                                ))
                            end
                        end
                    end
                end
                println("completed processing $member and $site")
            catch
                @info "Simulation failed for $member and $site. Appending NaNs..."
                push!(rows, (
                    member = member,
                    site = site,
                    variable = NaN,
                    statistic = NaN,
                    convection_type = convection_type,
                ))
            end
        end
    end

    return DataFrame(rows)
end

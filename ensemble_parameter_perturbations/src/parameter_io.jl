# Data loading given explicit paths.
#
# No dataset-specific glue and no config-schema assumptions here - every
# function takes an explicit path/rootdir and returns in-memory data. Reuse
# these directly if you have your own directory layout and just need "read
# this parameter.toml" / "read N members' parameter.tomls".
#
# Dataset-specific orchestration (the output_5_cfsites layout,
# experiment_config.yml schema, etc.) lives in experiment.jl.
#
# Expects the including script to have already done:
#   using TOML, DataFrames
#   import ClimaCalibrate as CAL
# and to have already `include`d methods.jl (for normalize_parameter_dict).

"""
    load_raw_parameter_toml(fpath)

Reads a `parameter.toml` file (as written by EnsembleKalmanProcesses /
ClimaCalibrate) and returns a `Dict` of raw (constrained-space) parameter
values, keyed by parameter name.
"""
function load_raw_parameter_toml(fpath)
    param_dict = TOML.parsefile(fpath)
    values_dict = Dict()
    for (key, value) in param_dict
        values_dict[key] = value["value"]
    end
    return values_dict
end

"""
    load_prior(prior_path)

Thin I/O wrapper around `ClimaCalibrate.get_prior`.
"""
load_prior(prior_path::String) = CAL.get_prior(prior_path)

"""
    load_member_parameters(rootdir, n_members; toml_name="parameter")

Reads `rootdir/member_NNN/\$(toml_name).toml` for `member in 1:n_members` and
returns a `Vector` of raw parameter `Dict`s, one per member, in member order.
"""
function load_member_parameters(rootdir::String, n_members::Int; toml_name::String = "parameter")
    param_dicts = []
    for member in 1:n_members
        member_dir = joinpath(rootdir, "member_" * lpad(member, 3, "0"))
        push!(param_dicts, load_raw_parameter_toml(joinpath(member_dir, "$(toml_name).toml")))
    end
    return param_dicts
end

"""
    normalize_parameters(fpath, param_stats)

Reads a member's `parameter.toml` from `fpath` and normalizes it against
precomputed per-parameter statistics (mean/std), returning a `Dict`.
"""
normalize_parameters(fpath::String, param_stats::DataFrame) = normalize_parameter_dict(load_raw_parameter_toml(fpath), param_stats)

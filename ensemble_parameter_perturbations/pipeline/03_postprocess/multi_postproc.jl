import EnsembleKalmanProcesses as EKP
import YAML
import ClimaCalibrate as CAL
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis
using DataFrames
import CSV
using FixedEffectModels
using TOML
using LinearAlgebra

# Get member number from command line argument
if length(ARGS) < 1
    error("Usage: julia multi_postproc.jl <member_number>")
end

member_num = parse(Int, ARGS[1])
member = "member_" * lpad(member_num, 3, '0')  # Format as member_001, member_002, etc.

@info "Processing for member $member"

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "methods.jl"))
include(joinpath(PROJECT_ROOT, "src", "parameter_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiment.jl"))

config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
resolve_config_paths!(config, PROJECT_ROOT)

param_stats_df = compute_parameter_statistics(config["prior_path"])

sites = Set(basename.(glob(config["output_dir"] * "/*/*")))
pop!(sites, "parameter.toml")
sites = collect(sites)

# Process only the specified member
df = process_members_sites([member], sites, param_stats_df, config)

# Create output directory if it doesn't exist
output_dir = joinpath(PROJECT_ROOT, "data", "postprocessing", config["exp_name"])
if !isdir(output_dir)
    mkpath(output_dir)
end

# Save dataframe for this member
output_file = joinpath(output_dir, "$(member).csv")
CSV.write(output_file, df)

@info "Completed processing member $member, saved to $output_file"

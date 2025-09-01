import EnsembleKalmanProcesses as EKP
import ScikitLearn
import YAML
import JLD2
import ClimaCalibrate as CAL
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis
using Revise
using DataFrames
import CSV
using ColorSchemes
import CalibrateEmulateSample as CES
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

include(joinpath(@__DIR__, "..", "helper_funcs.jl"))
config = YAML.load_file(joinpath(@__DIR__, "..", "experiment_config.yml"));

param_stats_df = compute_parameter_statistics(config["prior_path"])

sites = Set(basename.(glob(config["output_dir"] * "/*/*")))
pop!(sites, "parameter.toml")
sites = collect(sites)

# Process only the specified member
df = process_members_sites([member], sites, param_stats_df, config)

# Create output directory if it doesn't exist
output_dir = joinpath(@__DIR__, "dataframes", config["exp_name"])
if !isdir(output_dir)
    mkpath(output_dir)
end

# Save dataframe for this member
output_file = joinpath(output_dir, "$(member).csv")
CSV.write(output_file, df)

@info "Completed processing member $member, saved to $output_file"

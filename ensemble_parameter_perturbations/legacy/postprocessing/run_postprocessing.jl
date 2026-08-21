# Legacy serial postprocessing driver: processes ALL members in one pass and
# writes one big concatenated CSV. Superseded by pipeline/03_postprocess/
# multi_postproc.jl, which does the same per-member work as a SLURM array job
# and writes one CSV per member (much faster for ~50k simulations). Kept for
# reference / small-scale debugging.

import EnsembleKalmanProcesses as EKP
import YAML
import ClimaCalibrate as CAL
using Statistics
using Glob
using ClimaAnalysis
using DataFrames
import CSV
using FixedEffectModels
using TOML
using LinearAlgebra

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "methods.jl"))
include(joinpath(PROJECT_ROOT, "src", "parameter_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiment.jl"))

config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
resolve_config_paths!(config, PROJECT_ROOT)

# compute the normalization coefficients for the prior
param_stats_df = compute_parameter_statistics(config["prior_path"])

# get a list of the number of members and the sites that we are running for these experiments
members = basename.(glob(config["output_dir"] * "/*"))
sites = Set(basename.(glob(config["output_dir"] * "/*/*")))
pop!(sites, "parameter.toml")
sites = collect(sites)

# process the statistics for the dataframe
df = process_members_sites(members, sites, param_stats_df, config)

legacy_data_dir = joinpath(@__DIR__, "..", "data_dataframes")
mkpath(legacy_data_dir)
CSV.write(joinpath(legacy_data_dir, "$(config["exp_name"]).csv"), df)
println("completed processing dataframes")

# clean up dataframe and write cleaned version as well
df_cleaned = normalize_statistics(df)

CSV.write(joinpath(legacy_data_dir, "$(config["exp_name"])_cleaned.csv"), df_cleaned)

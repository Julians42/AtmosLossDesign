import Pkg; Pkg.activate(".")
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


include("helper_funcs.jl")
config = YAML.load_file("experiment_config.yml");

# compute the normalization coefficients for the prior
param_stats_df = compute_parameter_statistics(config["prior_path"])

# example usage:
# ex_norm_params = normalize_parameters("output_2/member_001/parameter.toml", param_stats_df)

# get a list of the number of members and the sites that we are running for these experiments
# eventually we'll want a dictionary of the total sites
members = basename.(glob(config["output_dir"] * "/*"))
sites = Set(basename.(glob(config["output_dir"] * "/*/*")))
pop!(sites, "parameter.toml")
sites = collect(sites)

# process the statistics for the dataframe
df = process_members_sites(members, sites, param_stats_df, config)

CSV.write("dataframes/$(config["exp_name"]).csv", df) 

# clean up dataframe and write cleaned version as well
df_cleaned = postprocess_dataframe(df)


CSV.write("dataframes/$(config["exp_name"])_cleaned.csv", df_cleaned)


# compute the B matrix for the cleaned dataframe
B_matrix, B_names = compute_regression_coefficients(df_cleaned, config)


# visualize
param_names = first(df_cleaned).keys
figs = plot_regression_analysis(B_matrix, B_names, param_names, config)


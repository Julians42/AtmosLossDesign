"""
evaluate.jl

Script to evaluate regression analysis from preprocessed dataframes.
Takes cleaned dataframes and produces regression coefficient analysis plots.
"""
# Ensure Pkg is available and activate environment
# using Pkg
# cd("/net/sampo/data1/jschmitt/AtmosLossDesign/ensemble_parameter_perturbations")
# Pkg.activate(".") 

import EnsembleKalmanProcesses as EKP
import YAML
import ClimaCalibrate as CAL
using Statistics
using CairoMakie
using DataFrames
using DataFramesMeta
import CSV
using LinearAlgebra
using FixedEffectModels
import TOML
using Glob

include("new_helper_funcs.jl")

# load the config
config = YAML.load_file("experiment_config.yml")

# read in all 100 dataframes into one concatenated dataframe
# df = CSV.read("dataframes/$(config["exp_name"]).csv", DataFrame)
df = @. CSV.read(glob("postprocessing/dataframes/diagnostic_edmfx/*.csv"), DataFrame) 

df = postprocess_dataframe(df)


# param_stats = compute_parameter_statistics(config["prior_path"])
# df_cleaned = postprocess_dataframe(df, param_stats, 100, "output_5_cfsites")

# compute regression coefficients for deep and shallow separately
# informing_variables_deep = get_all_variables(config, Config_cfsites_deep()) 
# informing_variables_shallow = get_all_variables(config, Config_cfsites_shallow()) 
# informing_variables = union(informing_variables_deep, informing_variables_shallow)
# reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables)
# # remove the failed regression variables from the list of names 
# informing_variables = informing_variables[setdiff(1:end, failed_indices)]

all_variables = get_all_variables(config, Config_cfsites_deep())
full_ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering = information_gain(all_variables, df, Linear())

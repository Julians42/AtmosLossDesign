"""
evaluate.jl

Script to evaluate regression analysis from preprocessed dataframes.
Takes cleaned dataframes and produces regression coefficient analysis plots.
"""
# Ensure Pkg is available and activate environment
using Pkg
cd("/net/sampo/data1/jschmitt/AtmosLossDesign/ensemble_parameter_perturbations")
Pkg.activate(".") 

import EnsembleKalmanProcesses as EKP
import YAML
import ClimaCalibrate as CAL
using Statistics
using CairoMakie
using DataFrames
import CSV
using LinearAlgebra
using FixedEffectModels
import TOML

include("helper_funcs.jl")

# load the config
config = YAML.load_file("experiment_config.yml")

# load and clean dataframe 
df = CSV.read("dataframes/$(config["exp_name"]).csv", DataFrame)
param_stats = compute_parameter_statistics(config["prior_path"])
df_cleaned = postprocess_dataframe(df, param_stats, 100, "../../ensemble_parameter_perturbations/output_2")

# compute regression coefficients
informing_variables = get_all_variables(config) # default is using all variables
reg_coefs, coef_names = compute_regression_coefficients(df_cleaned, informing_variables)



sorted_var_vals, sorted_var_inds, sorted_var_names = plot_regression_analysis(reg_coefs, informing_variables, coef_names, config["exp_name"])

include("utils/elbow_calculation.jl")
eigs = svd(reg_coefs*reg_coefs').S

println("Kneedle method: ", dimension_by_kneedle(eigs))
println("Second derivative method: ", elbow_second_derivative(eigs))
println("95% variance method: ", elbow_percentage_cutoff(eigs))

all_configs = Dict("full_variables" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi", "dsevi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "reduction_start_time" => 64800,
        "reduction_end_time" => 108000,
        "exp_name" => "full_variables",
        "prior_path" => "prior.toml"
    ),
    "realistic_variables" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "hur", "cl"],
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "reduction_start_time" => 64800,
        "reduction_end_time" => 108000,
        "exp_name" => "realistic_variables",
        "prior_path" => "prior.toml"
    ),
    "realistic_no_advanced_clouds" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs"],
        "var_names_prof" => ["ta", "hus", "hur"],
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "reduction_start_time" => 64800,
        "reduction_end_time" => 108000,
        "exp_name" => "realistic_no_advanced_clouds",
        "prior_path" => "prior.toml"
    ),
    "realistic_no_surface_levels" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "hur", "cl"],
        "z_levels" => [500, 1000, 1500, 2000, 2500, 3000, 4000],
        "reduction_start_time" => 64800,
        "reduction_end_time" => 108000,
        "exp_name" => "realistic_no_surface_levels",
        "prior_path" => "prior.toml"
    ),
    "realistic_plus_tke" => 
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "hur", "cl"],   
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "reduction_start_time" => 64800,
        "reduction_end_time" => 108000,
        "exp_name" => "realistic_plus_tke",
        "prior_path" => "prior.toml"
    )
)


for (key, value) in all_configs

    informing_variables = get_all_variables(value) # default is using all variables
    println(length(informing_variables), " variables")
    reg_coefs, coef_names = compute_regression_coefficients(df_cleaned, informing_variables);

    eigs = svd(reg_coefs*reg_coefs').S

    println(value["exp_name"])
    println("Kneedle method: ", dimension_by_kneedle(eigs))
    println("Second derivative method: ", elbow_second_derivative(eigs))
    println("95% variance method: ", elbow_percentage_cutoff(eigs))
    println("Number of eigs larger than 0.05: ", sum(eigs .> 0.05))
    println("--------------------------------")

    sorted_var_vals, sorted_var_inds, sorted_var_names = plot_regression_analysis(reg_coefs, informing_variables, coef_names, value["exp_name"])
end

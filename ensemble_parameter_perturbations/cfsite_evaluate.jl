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
using Glob

include("new_helper_funcs.jl")

# load the config
config = YAML.load_file("experiment_config.yml")

# read in all 100 dataframes into one concatenated dataframe
# df = CSV.read("dataframes/$(config["exp_name"]).csv", DataFrame)
df = @. CSV.read(glob("postprocessing/dataframes/diagnostic_edmfx/*.csv"), DataFrame) 
param_stats = compute_parameter_statistics(config["prior_path"])
df_cleaned = postprocess_dataframe(df, param_stats, 100, "output_5_cfsites")

# compute regression coefficients for deep and shallow separately
informing_variables_deep = get_all_variables(config, Config_cfsites_deep()) 
informing_variables_shallow = get_all_variables(config, Config_cfsites_shallow()) 
informing_variables = union(informing_variables_deep, informing_variables_shallow)
reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables)
# remove the failed regression variables from the list of names 
informing_variables = informing_variables[setdiff(1:end, failed_indices)]


sorted_var_vals, sorted_var_inds, sorted_var_names = plot_regression_analysis(reg_coefs, informing_variables, coef_names, config)

all_configs = Dict("full_variables" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi", "dsevi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => Dict("shallow" => [100, 100, 4000], "deep" => [100, 100, 10000]),
        "exp_name" => "full_variables",
    ),
    "realistic_variables" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "hur", "cl"],
        "z_levels" => Dict("shallow" => [100, 100, 4000], "deep" => [100, 100, 10000]),
        "exp_name" => "realistic_variables",
    ),
    "no_advanced_clouds" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs"],
        "var_names_prof" => ["ta", "hus", "hur"],
        "z_levels" => Dict("shallow" => [100, 100, 4000], "deep" => [100, 100, 10000]),
        "exp_name" => "realistic_no_advanced_clouds",
    ),
    "realistic_plus_tke" => 
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "hur", "cl"],   
        "z_levels" => Dict("shallow" => [100, 100, 4000], "deep" => [100, 100, 10000]),
        "exp_name" => "realistic_plus_tke",
    ),
    "integrated_clouds" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus"],
        "z_levels" => Dict("shallow" => [100, 100, 4000], "deep" => [100, 100, 10000]),
        "exp_name" => "integrated_clouds",
    ),
    "realistic_plus_entr" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "hur", "cl", "arup", "entr"],
        "z_levels" => Dict("shallow" => [100, 100, 4000], "deep" => [100, 100, 10000]),
        "exp_name" => "realistic_plus_entr",
    ),
    "realistic_no_surface_levels" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "hur", "cl"],
        "z_levels" => Dict("shallow" => [1000, 100, 4000], "deep" => [1000, 100, 10000]),
        "exp_name" => "realistic_no_surface_levels",
    )
)


for (key, value) in all_configs

    # Get variables for both deep and shallow convection separately
    informing_variables_deep_config = get_all_variables(value, Config_cfsites_deep())
    informing_variables_shallow_config = get_all_variables(value, Config_cfsites_shallow())
    informing_variables_config = union(informing_variables_deep_config, informing_variables_shallow_config)
    
    println("$(value["exp_name"]): $(length(informing_variables_config)) variables ($(length(informing_variables_deep_config)) deep + $(length(informing_variables_shallow_config)) shallow)")
    
    reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables_config);

    # remove the failed regression variables from the list of names 
    informing_variables_config = informing_variables_config[setdiff(1:end, failed_indices)]

    # eigs = svd(reg_coefs*reg_coefs').S

    # println(value["exp_name"])
    # println("Kneedle method: ", dimension_by_kneedle(eigs))
    # println("Second derivative method: ", elbow_second_derivative(eigs))
    # println("95% variance method: ", elbow_percentage_cutoff(eigs))
    # println("Number of eigs larger than 0.05: ", sum(eigs .> 0.05))
    # println("--------------------------------")

    sorted_var_vals, sorted_var_inds, sorted_var_names = plot_regression_analysis(reg_coefs, informing_variables_config, coef_names, value)
end







# integrate the profile variables by computing the trace of the submatrix of the regression coefficients
informing_variables_all = union(
    get_all_variables(all_configs["realistic_no_surface_levels"], Config_cfsites_deep()),
    get_all_variables(all_configs["realistic_no_surface_levels"], Config_cfsites_shallow())
)
reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables_all);
informing_variables = informing_variables_all[setdiff(1:end, failed_indices)]


var_informing_ar = []
for profile_var in all_configs["realistic_no_surface_levels"]["var_names_prof"]
    matching_indx = getindex.(split.(informing_variables, "_"), 1) .== profile_var
    # get variable informed array
    var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
    # println(profile_var, " ", tr(var_informed_ar))
    # println(profile_var, " ", sum(matching_indx))
    push!(var_informing_ar, [profile_var, tr(var_informed_ar)])
end

for single_var in all_configs["realistic_no_surface_levels"]["var_names_int"]
    matching_indx = getindex.(split.(informing_variables, "_"), 1) .== single_var
    # get variable informed array
    var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
    println(single_var, " ", var_informed_ar[1,1])
    println(single_var, " ", sum(matching_indx))
    push!(var_informing_ar, [single_var, var_informed_ar[1,1]])
end

# Convert the array to a proper DataFrame
var_informing_matrix = hcat(var_informing_ar...)
var_informing_df = DataFrame(
    var_name = var_informing_matrix[1, :],
    var_informing = var_informing_matrix[2, :]
)

# Sort by var_informing magnitude (descending)
sort!(var_informing_df, :var_informing, rev=true)

# Create bar plot of variable informing dimension
fig_var_informing = Figure(size = (800, 400))
ax_var_informing = Axis(fig_var_informing[1,1],
    xlabel = "Variable",
    ylabel = "Variable Informing Value",
    title = "Variable Informing Dimension (realistic_no_surface_levels)")

barplot!(ax_var_informing, 1:length(var_informing_df.var_informing), var_informing_df.var_informing)
ax_var_informing.xticks = (1:length(var_informing_df.var_name), var_informing_df.var_name)
ax_var_informing.xticklabelrotation = π/2
save("plots/realistic_no_surface_levels_var_informing.png", fig_var_informing)

# Create the variable informing plot using the helper function
plot_variable_informing_analysis(reg_coefs, informing_variables, 
                               all_configs["realistic_no_surface_levels"]["var_names_prof"],
                               all_configs["realistic_no_surface_levels"]["var_names_int"],
                               "realistic_no_surface_levels")
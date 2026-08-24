"""
evaluate.jl

Legacy regression-coefficient analysis: evaluates regression analysis from
preprocessed dataframes and produces regression-coefficient/SVD-based
diagnostic plots. Superseded by the information-gain approach - see
pipeline/04_analysis/ and pipeline/05_plots/ for the actively maintained
pipeline, and ../README.md for how the two compare. Kept for reference.
"""
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
using JLD2
using LaTeXStrings

PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "methods.jl"))
include(joinpath(PROJECT_ROOT, "src", "parameter_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiment.jl"))
include(joinpath(PROJECT_ROOT, "src", "plotting_theme.jl"))
include(joinpath(@__DIR__, "regression_funcs.jl"))
include(joinpath(@__DIR__, "elbow_calculation.jl"))

set_default_plot_theme!()

# load the config
config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
resolve_config_paths!(config, PROJECT_ROOT)

# load and clean dataframe
df = CSV.read(joinpath(@__DIR__, "data_dataframes", "$(config["exp_name"]).csv"), DataFrame)
param_stats = compute_parameter_statistics(config["prior_path"])
df_cleaned = postprocess_dataframe(df, param_stats, 100, config["output_dir"])

# compute regression coefficients
informing_variables = get_all_variables(config, Config_cfsites()) # default is using all variables
reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables)
# remove the failed regression variables from the list of names
informing_variables = informing_variables[setdiff(1:end, failed_indices)]

sorted_var_vals, sorted_var_inds, sorted_var_names = plot_regression_analysis(reg_coefs, informing_variables, coef_names, config)

eigs = svd(reg_coefs*reg_coefs').S

println("Kneedle method: ", dimension_by_kneedle(eigs))
println("Second derivative method: ", elbow_second_derivative(eigs))
println("95% variance method: ", elbow_percentage_cutoff(eigs))

all_configs = Dict("full_variables" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi", "dsevi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "exp_name" => "full_variables",
    ),
    "realistic_variables" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "hur", "cl"],
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "exp_name" => "realistic_variables",
    ),
    "realistic_no_advanced_clouds" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs"],
        "var_names_prof" => ["ta", "hus", "hur"],
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "exp_name" => "realistic_no_advanced_clouds",
    ),
    "realistic_no_surface_levels" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "hur", "cl"],
        "z_levels" => [500, 1000, 1500, 2000, 2500, 3000, 4000],
        "exp_name" => "realistic_no_surface_levels",
    ),
    "realistic_plus_tke" =>
    Dict(
        "var_names_int" => ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi"],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "hur", "cl"],
        "z_levels" => [100, 300, 500, 700, 850, 1000, 1150, 1300, 1500, 2000, 2500, 3000, 4000],
        "exp_name" => "realistic_plus_tke",
    )
)


for (key, value) in all_configs
    # all of these already exist as globals from the single-config run above -
    # without this declaration each would be treated as a new loop-local
    # variable (Julia's top-level "soft scope" ambiguity) instead of updating
    # the existing global.
    global informing_variables, reg_coefs, coef_names, failed_indices, eigs
    global sorted_var_vals, sorted_var_inds, sorted_var_names

    informing_variables = get_all_variables(value, Config_cfsites())
    println(length(informing_variables), " variables")
    reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables);

    # remove the failed regression variables from the list of names
    informing_variables = informing_variables[setdiff(1:end, failed_indices)]

    eigs = svd(reg_coefs*reg_coefs').S

    println(value["exp_name"])
    println("Kneedle method: ", dimension_by_kneedle(eigs))
    println("Second derivative method: ", elbow_second_derivative(eigs))
    println("95% variance method: ", elbow_percentage_cutoff(eigs))
    println("Number of eigs larger than 0.05: ", sum(eigs .> 0.05))
    println("--------------------------------")

    sorted_var_vals, sorted_var_inds, sorted_var_names = plot_regression_analysis(reg_coefs, informing_variables, coef_names, value)
end


# integrate the profile variables by computing the trace of the submatrix of the regression coefficients
informing_variables_all = get_all_variables(all_configs["realistic_no_surface_levels"], Config_cfsites())
reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables_all);
informing_variables = informing_variables_all[setdiff(1:end, failed_indices)]


var_informing_ar = []
for profile_var in all_configs["realistic_no_surface_levels"]["var_names_prof"]
    matching_indx = getindex.(split.(informing_variables, "_"), 1) .== profile_var
    # get variable informed array
    var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
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
save(joinpath(PROJECT_ROOT, "plots", "realistic_no_surface_levels_var_informing.png"), fig_var_informing)

# Create the variable informing plot using the helper function
plot_variable_informing_analysis(reg_coefs, informing_variables,
                               all_configs["realistic_no_surface_levels"]["var_names_prof"],
                               all_configs["realistic_no_surface_levels"]["var_names_int"],
                               "realistic_no_surface_levels")

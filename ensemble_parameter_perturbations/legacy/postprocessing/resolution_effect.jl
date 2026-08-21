# For the integrated variables we want to see how the variable informing
# integrated magnitude changes with resolution for the profile variables.
#
# Legacy regression-coefficient analysis; superseded by the information-gain
# approach (see pipeline/04_analysis/, pipeline/05_plots/). Kept for
# reference.

import YAML
import CSV
using DataFrames
import TOML
using Statistics
using CairoMakie
using Glob
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
using FixedEffectModels
using LinearAlgebra

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "methods.jl"))
include(joinpath(PROJECT_ROOT, "src", "parameter_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiment.jl"))
include(joinpath(PROJECT_ROOT, "src", "plotting_theme.jl"))
include(joinpath(@__DIR__, "..", "regression_funcs.jl"))
include(joinpath(@__DIR__, "information_metrics.jl"))

set_default_plot_theme!()

main_config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
resolve_config_paths!(main_config, PROJECT_ROOT)

legacy_data_dir = joinpath(PROJECT_ROOT, "legacy", "data_dataframes")

all_res_configs = Dict("res_1000" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [1000, 2000, 3000, 4000],
        "exp_name" => "res_1000",
    ),
    "res_500" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [500, 1000, 1500, 2000, 2500, 3000, 4000],
        "exp_name" => "res_500",
    ),
    "res_250" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000],
        "exp_name" => "res_250",
    ),
    "res_100" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000],
        "exp_name" => "res_100",
    ),
)

# compile all the needed z_levels
all_z_levels = unique(vcat([all_res_configs[key]["z_levels"] for key in keys(all_res_configs)]...))
sort!(all_z_levels)

# make the full config
full_config = Dict(
    "var_names_int" => [],
    "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
    "z_levels" => all_z_levels,
    "exp_name" => "resolution_effect",
    "prior_path" => main_config["prior_path"],
    "output_dir" => main_config["output_dir"],
    "toml_path_name" => main_config["toml_path_name"],
    "reduction_start_time" => main_config["reduction_start_time"],
    "reduction_end_time" => main_config["reduction_end_time"],
)

# get the dataframe
param_stats_df = compute_parameter_statistics(full_config["prior_path"])

# get a list of the number of members and the sites that we are running for these experiments
members = basename.(glob(full_config["output_dir"] * "/*"))
sites = Set(basename.(glob(full_config["output_dir"] * "/*/*")))
pop!(sites, "parameter.toml")
sites = collect(sites)

# df = process_members_sites(members, sites, param_stats_df, full_config)
# CSV.write(joinpath(legacy_data_dir, "resolution_effect.csv"), df)

df = CSV.read(joinpath(legacy_data_dir, "resolution_effect.csv"), DataFrame)

param_stats = compute_parameter_statistics(full_config["prior_path"])
# NOTE: "output_4_diagnostic_edmfx" predates the current output_5_cfsites
# experiment and does not exist on disk as of this reorg - verify/update
# before relying on this script.
df_cleaned = postprocess_dataframe(df, param_stats, 100, joinpath(PROJECT_ROOT, "output_4_diagnostic_edmfx"))

dfs = []
for (key, config_res) in all_res_configs

    # get the variables
    informing_variables = get_all_variables(config_res, Config_cfsites())
    reg_coefs, coef_names, failed_indices = compute_regression_coefficients_optimized(df_cleaned, informing_variables)
    # remove the failed regression variables from the list of names
    informing_variables = informing_variables[setdiff(1:end, failed_indices)]

    var_informing_df = analyze_variable_information(reg_coefs, informing_variables, config_res["var_names_prof"], config_res["var_names_int"], config_res["exp_name"], EffectiveRankMetric())

    push!(dfs, var_informing_df)
end

joined = reduce((df1, df2) -> outerjoin(df1, df2, on=:var_name), dfs)
df_long = stack(joined, Not(:var_name), variable_name=:resolution, value_name=:var_informing)

# Shorten resolution labels to just the numbers
df_long.resolution = replace.(df_long.resolution, "res_" => "")

# Sort the dataframe to ensure correct order of resolutions
res_order = ["100", "250", "500", "1000"]
df_long.resolution = parse.(Int, df_long.resolution)  # Convert to integers for proper sorting
sort!(df_long, [:resolution, :var_name])

# Convert resolution back to strings for plotting compatibility
df_long.resolution = string.(df_long.resolution)

# Create improved plot
f = Figure(size = (1000, 600))
ax = Axis(f[1,1],
    xlabel = "Variable",
    ylabel = "Information available per variable for calibration",
    title = "Effect of resolution on information available for calibration by variable (Effective Rank Metric)",
    xticklabelrotation = 45
)

# Get unique variables and resolutions for positioning
unique_vars = unique(df_long.var_name)
unique_resolutions = unique(df_long.resolution)

# Create numeric positions for variables
var_positions = Dict(var => i for (i, var) in enumerate(unique_vars))
x_positions = [var_positions[var] for var in df_long.var_name]

# Create dodge groups (numeric indices for each resolution)
res_positions = Dict(res => i for (i, res) in enumerate(unique_resolutions))
dodge_groups = [res_positions[res] for res in df_long.resolution]

# Create color map for resolutions
colors = [:blue, :red, :green, :orange]
color_map = Dict(res => colors[i] for (i, res) in enumerate(unique_resolutions))
bar_colors = [color_map[res] for res in df_long.resolution]

# Create bar plot with proper dodging
barplot!(ax, x_positions, df_long.var_informing,
    dodge = dodge_groups,
    color = bar_colors,
    width = 0.6,
    gap = 0.2
)

# Set x-axis labels
ax.xticks = (1:length(unique_vars), unique_vars)

# Add legend
Legend(f[1,2], [PolyElement(color = color_map[res]) for res in unique_resolutions],
       unique_resolutions, "Resolution (m)")

# Adjust layout
rowgap!(f.layout, 10)
colgap!(f.layout, 20)

save(joinpath(PROJECT_ROOT, "plots", "resolution_effect_effective_rank.png"), f, px_per_unit = 2)

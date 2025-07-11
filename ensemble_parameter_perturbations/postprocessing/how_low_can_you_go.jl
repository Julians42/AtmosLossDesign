# For the integrated variables we want to see how the variable informing integrated magnitude changes with resolution for the profile variables


import YAML
import CSV
using DataFrames
import TOML
using Statistics
using CairoMakie
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
using Glob

include("../helper_funcs.jl")

main_config = YAML.load_file("../experiment_config.yml")

df = CSV.read("../dataframes/$(main_config["exp_name"]).csv", DataFrame)


all_res_configs = Dict("res_1000" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [1000],
        "exp_name" => "res_1000",
    ),
    "res_500" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [500],
        "exp_name" => "res_500",
    ),
    "res_250" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [250],
        "exp_name" => "res_250",
    ),
    "res_100" =>
    Dict(
        "var_names_int" => [],
        "var_names_prof" => ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"],
        "z_levels" => [100],
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
# eventually we'll want a dictionary of the total sites
members = basename.(glob(full_config["output_dir"] * "/*"))
sites = Set(basename.(glob(full_config["output_dir"] * "/*/*")))
pop!(sites, "parameter.toml")
sites = collect(sites)

# df = process_members_sites(members, sites, param_stats_df, full_config)

# # save dataframe 
# CSV.write("dataframes/resolution_effect.csv", df)


df = CSV.read("dataframes/resolution_effect.csv", DataFrame)

param_stats = compute_parameter_statistics(full_config["prior_path"])
df_cleaned = postprocess_dataframe(df, param_stats, 100, "output_4_diagnostic_edmfx")


function get_variable_informing_analysis(reg_coefs, informing_variables, var_names_prof, var_names_int, exp_name)
    var_informing_ar = []

    # Process profile variables (integrate across levels)
    for profile_var in var_names_prof
        matching_indx = getindex.(split.(informing_variables, "_"), 1) .== profile_var
        # get variable informed array
        var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
        println(profile_var, " ", tr(var_informed_ar))
        println(profile_var, " ", sum(matching_indx))
        push!(var_informing_ar, [profile_var, tr(var_informed_ar)])
    end

        # Process integrated variables (single values)
    for single_var in var_names_int
        try
            matching_indx = getindex.(split.(informing_variables, "_"), 1) .== single_var
            # get variable informed array
            var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
            println(single_var, " ", var_informed_ar[1,1])
            println(single_var, " ", sum(matching_indx))
            push!(var_informing_ar, [single_var, var_informed_ar[1,1]])
        catch
            println(single_var, " failed")
        end
    end

    # Convert the array to a proper DataFrame
    var_informing_matrix = hcat(var_informing_ar...)
    var_informing_df = DataFrame([
        :var_name => var_informing_matrix[1, :],
        Symbol(exp_name) => var_informing_matrix[2, :]
    ])

    # Sort by var_informing magnitude (descending)
    # sorted_df = sort(var_informing_df, :var_informing, rev=true)
    # return sorted_df
    return var_informing_df
end

dfs = []
for (key, config_res) in all_res_configs

    # get the variables 
    informing_variables = get_all_variables(config_res)
    reg_coefs, coef_names, failed_indices = compute_regression_coefficients(df_cleaned, informing_variables)
    # remove the failed regression variables from the list of names 
    informing_variables = informing_variables[setdiff(1:end, failed_indices)]

    var_informing_df = get_variable_informing_analysis(reg_coefs, informing_variables, config_res["var_names_prof"], config_res["var_names_int"], config_res["exp_name"])

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
    title = "Information content at single level (e.g. measuring how close to the surface we can measure)",
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
       unique_resolutions, "Measurement level (m)")

# Adjust layout
rowgap!(f.layout, 10)
colgap!(f.layout, 20)

save("plots/how_low_can_you_go.png", f, px_per_unit = 2)

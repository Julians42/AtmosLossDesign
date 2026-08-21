# Compute the parameter informedness for several different cases 
# E.g. study the impact of having unseen variables, lower resolution, etc on parameter informedness 
# and total information gain.

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
using CategoricalArrays

include("new_helper_funcs.jl")

# scenario configurations to test
realistic_integrated_vars = ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi", "dsevi"]
realistic_no_advanced_clouds_profile_vars = ["ta", "hus", "hur"]
realistic_profile_vars = ["ta", "hus", "clw", "cli", "hur", "cl"]
realistic_profile_vars_with_tke = ["ta", "hus", "clw", "cli", "tke", "hur", "cl"]
realistic_profile_vars_with_tke_and_entr = ["ta", "hus", "clw", "cli", "tke", "hur", "cl", "entr"]
realistic_profile_vars_with_tke_and_entr_and_arup = ["ta", "hus", "clw", "cli", "tke", "hur", "cl", "entr", "arup"]
low_resolution = [1000, 1000, 4000]
medium_resolution = [500, 500, 4000]
high_resolution = [100, 100, 4000]

# load the config
config = YAML.load_file("experiment_config.yml")
all_variables = get_all_variables(config, Config_cfsites_deep())

@load "bootstrap_sites/no_bootstrap/results.jld2" full_ig ∇G Σ_y Σ_0 constrained_params param_ordering

realistic_vars = create_var_list(realistic_profile_vars, realistic_integrated_vars, medium_resolution)
realistic_no_cloud_profiles_vars = create_var_list(realistic_no_advanced_clouds_profile_vars, realistic_integrated_vars, medium_resolution)
realistic_profile_vars_with_tke_vars = create_var_list(realistic_profile_vars_with_tke, realistic_integrated_vars, medium_resolution)
realistic_profile_vars_with_tke_and_entr_vars = create_var_list(realistic_profile_vars_with_tke_and_entr, realistic_integrated_vars, medium_resolution)
realistic_profile_vars_with_tke_and_entr_and_arup_vars = create_var_list(realistic_profile_vars_with_tke_and_entr_and_arup, realistic_integrated_vars, medium_resolution)

# parameter informedness
sub_pi = subset_parameter_informedness(realistic_vars, all_variables, Σ_y, Σ_0, ∇G)
sub_pi_no_cloud_profiles = subset_parameter_informedness(realistic_no_cloud_profiles_vars, all_variables, Σ_y, Σ_0, ∇G)
sub_pi_profile_vars_with_tke = subset_parameter_informedness(realistic_profile_vars_with_tke_vars, all_variables, Σ_y, Σ_0, ∇G)
sub_pi_profile_vars_with_tke_and_entr = subset_parameter_informedness(realistic_profile_vars_with_tke_and_entr_vars, all_variables, Σ_y, Σ_0, ∇G)
sub_pi_profile_vars_with_tke_and_entr_and_arup = subset_parameter_informedness(realistic_profile_vars_with_tke_and_entr_and_arup_vars, all_variables, Σ_y, Σ_0, ∇G)

t = subset_parameter_informedness(all_variables, all_variables, Σ_y, Σ_0, ∇G)

# put in a dataframe 
param_informedness_df = DataFrame(parameter = param_ordering, 
                                  hires_all = t,
                                  realisticw_tke_entr_arup = sub_pi_profile_vars_with_tke_and_entr_and_arup,
                                  realistic_w_tke_entr = sub_pi_profile_vars_with_tke_and_entr, 
                                  realistic_w_tke = sub_pi_profile_vars_with_tke, 
                                  realistic = sub_pi, 
                                  realistic_no_cloud_profiles = sub_pi_no_cloud_profiles, 
)
param_informedness_df = @orderby(param_informedness_df, -:hires_all)

# editable short-name map for long parameter names
short_name_map = Dict(
    "mixing_length_tke_surf_flux_coeff"        => L"l_{\mathrm{tke, surf}}",
    "mixing_length_diss_coeff"                 => L"D_\kappa",
    "precipitation_timescale"                  => L"\tau_p",
    "entr_param_vec_1"                         => L"\Pi_1",
    "entr_param_vec_2"                         => L"\Pi_2",
    "entr_param_vec_3"                         => L"\Pi_3",
    "entr_param_vec_4"                         => L"\Pi_4",
    "entr_param_vec_5"                         => L"\Pi_5",
    "entr_param_vec_6"                         => L"\Pi_6",
    "entr_inv_tau"                             => L"\tau_{\mathrm{entr}}^{-1}",
    "EDMF_surface_area"                        => L"a_{\mathrm{up,surf}}",
    "mixing_length_eddy_viscosity_coefficient" => L"l_{\mathrm{ev}}",
    "pressure_normalmode_drag_coeff"           => L"P_d",
    "mixing_length_Prandtl_number_0"           => L"\mathrm{Pr}_0",
    "entr_mult_limiter_coeff"                  => L"\lambda_{\mathrm{entr}}",
    "mixing_length_static_stab_coeff"          => L"l_{\mathrm{stab}}",
    "pressure_normalmode_buoy_coeff1"          => L"b_1",
)
scenario_name_map = Dict(
    "realistic" => "Observed variables at 500m resolution",
    "realistic_no_cloud_profiles" => "No cloud profiles",
    "realistic_w_tke" => "With TKE",
    "realistic_w_tke_entr" => "With TKE and Entrainment",
    "realisticw_tke_entr_arup" => "With TKE, Entrainment, and Updraft Area Fraction",
    "hires_all" => "All variables at 100m resolution",
)
param_informedness_df.short_name = [get(short_name_map, s, s) for s in param_informedness_df.parameter]

df_long = stack(param_informedness_df, 
                Not(["parameter", "short_name"]), 
                variable_name = :scenario, 
                value_name = :learnability
)

# make plot for the parameter informedness by variable 
with_theme(theme_minimal(), fontsize = 20) do
    fig = Figure(size = (1200, 600))

    groups = unique(df_long.short_name)
    scenarios = unique(df_long.scenario)
    colors = Makie.wong_colors()

    ax = Axis(fig[1, 1], 
        xlabel = "Parameter", 
        ylabel = "Parameter Informedness",
        # xticklabelrotation = π/4,
        xticks = (1:length(groups), groups),
        #title = "Parameter Informedness by Experiment",
        xticklabelsize = 20, 
        yticklabelsize = 20,
    )
    ylims!(ax, 0, nothing)

    # barplot!(ax, df_long.short_name, 
    barplot!(ax, map(s -> findfirst(isequal(s), groups), df_long.short_name), 
            df_long.learnability,
            # dodge = string.(df_long.scenario),
            # color = string.(df_long.scenario),
            dodge = Int.(map(s -> findfirst(isequal(s), scenarios), df_long.scenario)),
            color = colors[Int.(map(s -> findfirst(isequal(s), scenarios), df_long.scenario))],
            # strokewidth = 1,
            # width = 1,
    )

    labels = [get(scenario_name_map, s, s) for s in scenarios]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = "Experiment"
    Legend(fig[1,1], elements, labels, title,
        tellheight = false,
        tellwidth = false,
        margin = (10, 10, 10, 10),
        halign = :right, valign = :top,
        # orientation = :horizontal,
    )

    save("plots/experiment_parameter_informedness.png", fig)
end

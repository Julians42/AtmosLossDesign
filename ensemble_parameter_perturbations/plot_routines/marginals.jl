# Quantify the marginal information gain of each variable
# and the total parameter informedness

# load and compute the full variables
# include("../cfsite_evaluate.jl")

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

include("new_helper_funcs.jl")

# load the config
config = YAML.load_file("experiment_config.yml")
all_variables = get_all_variables(config, Config_cfsites_deep())

@load "bootstrap_sites/no_bootstrap/results.jld2" full_ig ∇G Σ_y Σ_0 constrained_params param_ordering

marginal_info_gain = DataFrame(var = [], ig = [])
for variable in ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi", "dsevi"]
    single_var_config = Dict("var_names_int" => [variable], 
                            "var_names_prof" => [], 
                            "z_levels" => Dict("shallow" => [100, 100, 4000], 
                                               "deep" => [100, 100, 4000]), 
                            "exp_name" => "single_var_$(variable)")
    single_var_levels = get_all_variables(single_var_config, Config_cfsites_deep())
    sub_ig = subset_info_gain(single_var_levels, all_variables, Σ_y, Σ_0, ∇G)
    println("$(variable): $(sub_ig)")
    push!(marginal_info_gain, (var = variable, ig = sub_ig))
end

for variable in ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"]
    single_var_config = Dict("var_names_int" => [], 
                            "var_names_prof" => [variable], 
                            "z_levels" => Dict("shallow" => [100, 100, 4000], 
                                               "deep" => [100, 100, 4000]), 
                            "exp_name" => "single_var_$(variable)")
    single_var_levels = get_all_variables(single_var_config, Config_cfsites_deep())
    sub_ig = subset_info_gain(single_var_levels, all_variables, Σ_y, Σ_0, ∇G)
    println("$(variable): $(sub_ig)")
    push!(marginal_info_gain, (var = variable, ig = sub_ig))
end
marginal_info_gain = @orderby(marginal_info_gain, -:ig) # order by information gain


# parameter informedness
sub_pi = subset_parameter_informedness(all_variables, all_variables, Σ_y, Σ_0, ∇G)
param_df = DataFrame(parameter = param_ordering, learnability = sub_pi)
param_df = @orderby(param_df, -:learnability) # order by parameter learnability

# editable short-name map for long parameter names
short_name_map = Dict(
    "mixing_length_tke_surf_flux_coeff" => "ml_tke_flux",
    "mixing_length_diss_coeff" => "ml_diss",
    "precipitation_timescale" => "precip_tau",
    "entr_param_vec_1" => "entr1",
    "entr_param_vec_2" => "entr2",
    "entr_param_vec_3" => "entr3",
    "entr_param_vec_4" => "entr4",
    "entr_param_vec_5" => "entr5",
    "entr_param_vec_6" => "entr6",
    "EDMF_surface_area" => "EDMF_area",
    "mixing_length_eddy_viscosity_coefficient" => "ml_eddy_visc",
    "pressure_normalmode_drag_coeff" => "p_drag",
    "mixing_length_Prandtl_number_0" => "Pr0",
    "entr_mult_limiter_coeff" => "entr_lim",
    "entr_inv_tau" => "entr_inv_tau",
    "mixing_length_static_stab_coeff" => "ml_static_stab",
    "pressure_normalmode_buoy_coeff1" => "p_buoy1",
)
param_df.short_name = [get(short_name_map, s, s) for s in param_df.parameter]

################################################################################
fig = Figure(size = (1000, 500))

# variable IG (vertical bars)
ax = Axis(fig[1, 1]; xticklabelrotation = π/2, ylabel = "Single Variable Information Gain", xlabel = "Variable",
          xlabelsize = 16, ylabelsize = 16, xticklabelsize = 12, yticklabelsize = 12)
xs = 1:length(vals_sorted)
barplot!(ax, xs, vals_sorted; color = :steelblue, alpha = 1., strokecolor= :black, strokewidth = 2)
ax.xticks = (xs, names_sorted)

# parameter informedness (horizontal bars with short labels)
ax2 = Axis(fig[1, 2]; xticklabelrotation = π/2, xlabel = "Parameter", ylabel = "Normalized Full Sample Parameter Informedness",
           xlabelsize = 16, ylabelsize = 16, xticklabelsize = 11, yticklabelsize = 11)

xs2 = 1:length(sub_pi_sorted)
barplot!(ax2, xs2, sub_pi_sorted / norm(sub_pi_sorted, 1); color = :seagreen, alpha = 1., strokecolor= :black, strokewidth = 2)
ax2.xticks = (xs2, param_labels_short)

save("plots/marginal_info_gain_full_resolution.png", fig)

################################################################################
# Quantify uncertainty using the bootstrap sites
# uncertainty_df = DataFrame(var=String[], ig=Float64[], bootstrap = Int[])
uncertainty_df = CSV.read("marginals_bootstrap.csv", DataFrame)

for boot_i in 50:100
    @load "bootstrap_sites/bootstrap_$(boot_i)/results.jld2" full_ig ∇G Σ_y Σ_0 constrained_params param_ordering
    # integrated variables
    for variable in ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi", "dsevi"]
        single_var_config = Dict("var_names_int" => [variable], 
                                "var_names_prof" => [], 
                                "z_levels" => Dict("shallow" => [100, 100, 4000], 
                                                   "deep" => [100, 100, 4000]), 
                                "exp_name" => "single_var_$(variable)")
        single_var_levels = get_all_variables(single_var_config, Config_cfsites_deep())
        sub_ig = subset_info_gain(single_var_levels, all_variables, Σ_y, Σ_0, ∇G)
        push!(uncertainty_df, (var=variable, ig=sub_ig, bootstrap = boot_i))
    end

    # profile variables
    for variable in ["ta", "hus", "clw", "cli", "tke", "wa", "hur", "cl", "arup", "entr"]
        single_var_config = Dict("var_names_int" => [], 
                                "var_names_prof" => [variable], 
                                "z_levels" => Dict("shallow" => [100, 100, 4000], 
                                                   "deep" => [100, 100, 4000]), 
                                "exp_name" => "single_var_$(variable)")
        single_var_levels = get_all_variables(single_var_config, Config_cfsites_deep())
        sub_ig = subset_info_gain(single_var_levels, all_variables, Σ_y, Σ_0, ∇G)
        push!(uncertainty_df, (var=variable, ig=sub_ig, bootstrap = boot_i))
    end
    @info "Processed bootstrap $boot_i"
end

CSV.write("marginals_bootstrap.csv", uncertainty_df)


################################################################################
# Get the uncertainty of the parameter informedness 
param_informedness_uncertainty = DataFrame(parameter=String[], informedness=Float64[], bootstrap=Int[])
for boot_i in 1:100
    @load "bootstrap_sites/bootstrap_$(boot_i)/results.jld2" full_ig ∇G Σ_y Σ_0 constrained_params param_ordering

    sub_pi = subset_parameter_informedness(all_variables, all_variables, Σ_y, Σ_0, ∇G)

    mini_df = DataFrame(parameter = param_ordering, informedness = sub_pi, bootstrap = repeat([boot_i], length(param_ordering)))

    param_informedness_uncertainty = vcat(param_informedness_uncertainty, mini_df)
end
CSV.write("param_informedness_uncertainty.csv", param_informedness_uncertainty)

################################################################################
# Plot the marginal information gain with error bars from the bootstrap sites 

uncertainty_df = CSV.read("marginals_bootstrap.csv", DataFrame)
param_informedness_uncertainty = CSV.read("param_informedness_uncertainty.csv", DataFrame)


ig_uq_agg = combine(groupby(uncertainty_df, :var), :ig => mean => :ig_mean, 
                                             :ig => std => :ig_std,
                                             :ig => (x -> quantile(x, 0.05)) => :ig_q05,
                                             :ig => (x -> quantile(x, 0.95)) => :ig_q95)

agg_param = combine(groupby(param_informedness_uncertainty, :parameter), 
                                             :informedness => mean => :informedness_mean, 
                                             :informedness => std => :informedness_std,
                                             :informedness => (x -> quantile(x, 0.05)) => :informedness_q05,
                                             :informedness => (x -> quantile(x, 0.95)) => :informedness_q95)

marginal_ig_uq = innerjoin(marginal_info_gain, ig_uq_agg, on = :var)
marginal_ig_uq = @orderby(marginal_ig_uq, on = -:ig)

parameter_uq = innerjoin(param_df, agg_param, on = :parameter)
parameter_uq = @orderby(parameter_uq, on = -:learnability)


fig = Figure(size = (1200, 400))

ax = Axis(fig[1, 1]; 
        xticklabelrotation = π/2, 
        ylabel = "Single Variable Information Gain", 
        xlabel = "Variable",
        xlabelsize = 16, 
        ylabelsize = 16, 
        xticklabelsize = 12, 
        yticklabelsize = 12
)

barplot!(ax, 
        1:length(marginal_ig_uq.ig), 
        marginal_ig_uq.ig; 
        color = :steelblue, 
        alpha = 1., 
        strokecolor=:black, 
        strokewidth = 2
)
errorbars!(ax, 
        1:length(marginal_ig_uq.ig), 
        marginal_ig_uq.ig, 
        marginal_ig_uq.ig_std .* 1.96,  
        marginal_ig_uq.ig_std .* 1.96; 
        color = :black, 
        whiskerwidth = 10
)
ax.xticks = (1:length(marginal_ig_uq.ig), marginal_ig_uq.var)


# parameter informedness (horizontal bars with short labels)
ax2 = Axis(fig[1, 2]; 
        xticklabelrotation = π/2, 
        xlabel = "Parameter", 
        ylabel = "Normalized Parameter Learnability",
        xlabelsize = 16, 
        ylabelsize = 16, 
        xticklabelsize = 11, 
        yticklabelsize = 11
)

barplot!(ax2, 
        1:length(parameter_uq.learnability), 
        parameter_uq.learnability; 
        color = :seagreen, 
        alpha = 1., 
        strokecolor= :black, 
        strokewidth = 2
)

errorbars!(ax2, 
        1:length(parameter_uq.learnability), 
        parameter_uq.learnability, 
        parameter_uq.informedness_std .* 1.96,
        parameter_uq.informedness_std .* 1.96;
        color = :black, 
        whiskerwidth = 10
)

ax2.xticks = (1:length(parameter_uq.parameter), parameter_uq.short_name)

save("plots/marginal_info_gain_full_resolution_with_uncertainty.png", fig)

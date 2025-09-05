# Quantify the marginal information gain of each variable
# and the total parameter informedness

# load and compute the full variables
# include("../cfsite_evaluate.jl")

marginal_info_gain = Dict()
for variable in ["pr", "rlut", "rsut", "rsutcs", "rlutcs", "lwp", "clvi", "clivi", "dsevi"]
    single_var_config = Dict("var_names_int" => [variable], 
                            "var_names_prof" => [], 
                            "z_levels" => Dict("shallow" => [100, 100, 4000], 
                                               "deep" => [100, 100, 4000]), 
                            "exp_name" => "single_var_$(variable)")
    single_var_levels = get_all_variables(single_var_config, Config_cfsites_deep())
    sub_ig = subset_info_gain(single_var_levels, all_variables, Σ_y, Σ_0, ∇G)
    println("$(variable): $(sub_ig)")
    marginal_info_gain[variable] = sub_ig
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
    marginal_info_gain[variable] = sub_ig
end


# parameter informedness
sub_pi = subset_parameter_informedness(all_variables, all_variables, Σ_y, Σ_0, ∇G)

# sort variable-level IG (names and values)
sorted = sort(collect(pairs(marginal_info_gain)); by = last, rev = true)
names_sorted = first.(sorted)
vals_sorted = last.(sorted)

# sort parameter informedness consistently with labels
param_idx = sortperm(sub_pi; rev = true)
sub_pi_sorted = sub_pi[param_idx]
param_labels_sorted = param_ordering[param_idx]

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
param_labels_short = [get(short_name_map, s, s) for s in param_labels_sorted]

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


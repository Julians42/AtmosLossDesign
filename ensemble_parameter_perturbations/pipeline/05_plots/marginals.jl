# Quantify the marginal information gain of each variable
# and the total parameter informedness

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

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "methods.jl"))
include(joinpath(PROJECT_ROOT, "src", "parameter_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiment.jl"))

# load the config
config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
resolve_config_paths!(config, PROJECT_ROOT)
all_variables = get_all_variables(config, Config_cfsites_deep())

@load joinpath(PROJECT_ROOT, "data", "bootstrap_sites", "no_bootstrap", "results.jld2") full_ig ∇G Σ_y Σ_0 constrained_params param_ordering

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
param_df.short_name = [get(short_name_map, s, s) for s in param_df.parameter]

################################################################################
# Quantify uncertainty using the bootstrap sites. `marginals_bootstrap.csv` and
# `param_informedness_uncertainty.csv` are resumable caches: each loop below
# only computes the bootstrap indices in `bootstrap_range` that are NOT
# already present in the cache's `:bootstrap` column, so a re-run against a
# cache that already covers `bootstrap_range` skips the (slow) loop entirely
# rather than recomputing it. Delete a cache file to force a full recompute
# of that piece.
cache_dir = joinpath(PROJECT_ROOT, "data", "cache")
mkpath(cache_dir)
marginals_bootstrap_csv = joinpath(cache_dir, "marginals_bootstrap.csv")
param_informedness_csv = joinpath(cache_dir, "param_informedness_uncertainty.csv")

bootstrap_range = 1:100

uncertainty_df = isfile(marginals_bootstrap_csv) ? CSV.read(marginals_bootstrap_csv, DataFrame) : DataFrame(var=String[], ig=Float64[], bootstrap=Int[])
missing_boot_i = setdiff(bootstrap_range, unique(uncertainty_df.bootstrap))

if isempty(missing_boot_i)
    @info "marginals_bootstrap.csv already covers bootstrap $(bootstrap_range) - skipping"
else
    @info "Computing marginal info gain for bootstrap indices $(missing_boot_i)"
    for boot_i in missing_boot_i
        # `full_ig` etc. already exist as globals from the @load above - without
        # this declaration, @load's assignments would be treated as new locals
        # scoped to this loop (Julia's top-level "soft scope" ambiguity) and
        # silently discarded once the loop ends.
        global full_ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering
        @load joinpath(PROJECT_ROOT, "data", "bootstrap_sites", "bootstrap_$(boot_i)", "results.jld2") full_ig ∇G Σ_y Σ_0 constrained_params param_ordering
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
    CSV.write(marginals_bootstrap_csv, uncertainty_df)
end

################################################################################
# Get the uncertainty of the parameter informedness
param_informedness_uncertainty = isfile(param_informedness_csv) ? CSV.read(param_informedness_csv, DataFrame) : DataFrame(parameter=String[], informedness=Float64[], bootstrap=Int[])
missing_boot_i_pi = setdiff(bootstrap_range, unique(param_informedness_uncertainty.bootstrap))

if isempty(missing_boot_i_pi)
    @info "param_informedness_uncertainty.csv already covers bootstrap $(bootstrap_range) - skipping"
else
    @info "Computing parameter informedness for bootstrap indices $(missing_boot_i_pi)"
    for boot_i in missing_boot_i_pi
        # see the comment on the identical `global` declaration above
        global full_ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering, sub_pi, param_informedness_uncertainty
        @load joinpath(PROJECT_ROOT, "data", "bootstrap_sites", "bootstrap_$(boot_i)", "results.jld2") full_ig ∇G Σ_y Σ_0 constrained_params param_ordering

        sub_pi = subset_parameter_informedness(all_variables, all_variables, Σ_y, Σ_0, ∇G)

        mini_df = DataFrame(parameter = param_ordering, informedness = sub_pi, bootstrap = repeat([boot_i], length(param_ordering)))

        param_informedness_uncertainty = vcat(param_informedness_uncertainty, mini_df)
    end
    CSV.write(param_informedness_csv, param_informedness_uncertainty)
end

################################################################################
# Plot the marginal information gain with error bars from the bootstrap sites
@info "Plotting..."
uncertainty_df = CSV.read(marginals_bootstrap_csv, DataFrame)
param_informedness_uncertainty = CSV.read(param_informedness_csv, DataFrame)

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
        ylabel = "Parameter Uncertainty Reduction (%)       ",
        xlabel = "Variable Observed",
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xgridvisible = false,
        ygridvisible = false,
)
scale = 100
barplot!(ax,
        1:length(marginal_ig_uq.ig),
        marginal_ig_uq.ig * scale;
        color = :steelblue,
        alpha = 1.,
        strokecolor=:black,
        strokewidth = 2
)
errorbars!(ax,
        1:length(marginal_ig_uq.ig),
        marginal_ig_uq.ig * scale,
        marginal_ig_uq.ig_std .* 1.96 .* scale,
        marginal_ig_uq.ig_std .* 1.96 .* scale;
        color = :black,
        whiskerwidth = 10
)
ax.xticks = (1:length(marginal_ig_uq.ig), marginal_ig_uq.var)
hidespines!(ax, :t, :r)

# parameter informedness (horizontal bars with short labels)
ax2 = Axis(fig[1, 2];
        xlabel = "Parameter",
        ylabel = "Learnability",
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xgridvisible = false,
        ygridvisible = false,
)
hidespines!(ax2, :t, :r)

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

save(joinpath(PROJECT_ROOT, "plots", "marginal_info_gain_full_resolution_with_uncertainty.png"), fig)

# compute the reduction in variance associated with observing a set of observations 

import YAML
import JLD2
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis
using DataFrames
using CSV
using NaNStatistics
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
import TOML
using DataFramesMeta
using LinearAlgebra
using JLD2
using StatsPlots
using CategoricalArrays
using Measures

using Revise
includet("var_helper_funcs.jl")

constrained_params, params_ordered = constrained_and_normalized_parameters(;
                                    rootdir = "../ensemble_parameter_perturbations/data/output_5_cfsites",
                                    prior_path = "../ensemble_parameter_perturbations/config/priors/prior_diagnostic_pi_entr_smooth_entr_detr_coarse_amip_new.toml")
# probably better to eventually get it directly from here:
# prior = CAL.get_prior("tomls/prior_diagnostic_pi_entr_smooth_entr_detr_coarse_amip_new.toml")
Σ₀ = cov(constrained_params')


df = CSV.read("dataframes/spp1_cfsites_intersect2.csv", DataFrame)

grad_df = g(params_ordered)
sites = unique(grad_df.site)

# filter to low latitudes for now only
t = transform(grad_df, :site => ByRow(x -> split(x, "_")) => [:lat, :lon, :site_date])
transform!(t, [:lat, :lon, :site_date] => ByRow((lat, lon, site_date) -> (parse(Float64, lat), parse(Float64, lon), convert(String, site_date))) => [:lat, :lon, :site_date])
filter!(row -> (abs(row.lat) <= 30.0), t)

# grad vectors 
gs_vectors = []
for site in sites
    grad_site = gs(site, params_ordered)
    push!(gs_vectors, grad_site)
end

mat = hcat(gs_vectors...) 
mat[mat .== 0] .= NaN
mean_gs = vec(nanmean(mat, dims = 2))

# compute prior variance 
prior_var = mean_gs' * Σ₀ * mean_gs

# compute the prior CRE variance associated with each site 
# prior_vars = []
# for site in sites 
#     # failures are set to 0 which happens a lot
#     grad_site = gs(site, params_ordered)
#     println(grad_site)
#     push!(prior_vars, grad_site' * Σ₀ * grad_site)
# end


# Get the observational covariance matrix for a collection of observations
df = CSV.read("grad_q_data.csv", DataFrame)
df = df[abs.(parse.(Float64, convert.(String, first.(split.(df.site, "_"))))) .<= 30.0, :]

config = YAML.load_file("experiment_config.yml");

# for 100 meter resolution
var_reduction_100 = get_variance_reduction(
    config["var_names_prof"],
    config["var_names_int"],
    [100, 100, 4000],
    Σ₀,
    df,
    params_ordered,
    gs_vectors
)
# for 500 meter resolution
var_reduction_500 = get_variance_reduction(
    config["var_names_prof"],
    config["var_names_int"],
    [500, 500, 4000],
    Σ₀,
    df,
    params_ordered,
    gs_vectors
)

@df var_reduction_100 StatsPlots.boxplot(
    :variable,
    :reduction,
    group = :variable,
    color = :cornflowerblue,
    ylabel = "Fractional Uncertainty Reduction in ΔCRE",
    xlabel = "Observed Variable",
    legend = false,
    grid = false,
    dpi = 300,
    size = (850, 500),
    xtickfontsize = 10, 
    ytickfontsize = 10,
    legend_font_pointsize = 11,
    yguidefontsize = 12,
    xguidefontsize = 12,
    left_margin = 2mm,
)
savefig("figures/grads/variance_reduction_100m.png")

@df var_reduction_500 StatsPlots.boxplot(
    :variable,
    :reduction,
    group = :variable,
    color = :cornflowerblue,
    # xticks = (1:length(ordered_vars), ordered_vars),
    ylabel = "Fractional Uncertainty Reduction in ΔCRE",
    xlabel = "Observed Variable",
    legend = false,
    grid = false,
    dpi = 300,
    size = (850, 500),
    xtickfontsize = 10, 
    ytickfontsize = 10,
    legend_font_pointsize = 11,
    yguidefontsize = 12,
    xguidefontsize = 12,
    left_margin = 2mm,
)
savefig("figures/grads/variance_reduction_500m.png")



# For specific combinations of variables we compute the joint effect on variance reduction
vars = create_var_list(config["var_names_prof"], config["var_names_int"], [100, 100, 4000])#config["z_levels"]["shallow"])
Σ_θ_post_0K, Σ_θ_post_4K = observational_posterior_covariance(Σ₀, vars, df, params_ordered)
Σ_θ_post = Σ_θ_post_0K #(Σ_θ_post_0K + Σ_θ_post_4K) / 2.0
boot_gs_means = get_boot_cre_grad(gs_vectors, 1000)
Σ_cre_pre_bs = [mean_gs' * Σ₀ * mean_gs for mean_gs in boot_gs_means]
Σ_cre_post_bs = [mean_gs' * Σ_θ_post * mean_gs for mean_gs in boot_gs_means]
println(var, "  ", mean((Σ_cre_post_bs .- Σ_cre_pre_bs) ./ Σ_cre_pre_bs))
# var  -0.7495143407141623


vars = create_var_list(config["var_names_prof"], config["var_names_int"], [500, 500, 4000])#config["z_levels"]["shallow"])
Σ_θ_post_0K, Σ_θ_post_4K = observational_posterior_covariance(Σ₀, vars, df, params_ordered)
Σ_θ_post = Σ_θ_post_0K #(Σ_θ_post_0K + Σ_θ_post_4K) / 2.0
boot_gs_means = get_boot_cre_grad(gs_vectors, 1000)
Σ_cre_pre_bs = [mean_gs' * Σ₀ * mean_gs for mean_gs in boot_gs_means]
Σ_cre_post_bs = [mean_gs' * Σ_θ_post * mean_gs for mean_gs in boot_gs_means]
println(var, "  ", mean((Σ_cre_post_bs .- Σ_cre_pre_bs) ./ Σ_cre_pre_bs))
# var  -0.4631032990110164










vars = ["ta_100"]
# vars = create_var_list(["ta"], [], config["z_levels"]["shallow"])


df_clean = @subset(df, isfinite.(:low_0K) .& isfinite.(:high_0K) .& isfinite.(:low_4K) .& isfinite.(:high_4K))
df_clean = stack(df, [:low_0K, :high_0K, :low_4K, :high_4K], variable_name = :scenario, value_name = :value)
df_clean = @subset(df_clean, isfinite.(:value) .& (:observation .∈ Ref(vars)))

# for variance we want to remove the variation between sites first before calculating the covariance matrix
transform!(groupby(df_clean, [:site, :observation, :scenario]),
    :value => (x -> x .- mean(x)) => :value
)
wide = unstack(df_clean, [:param, :site, :scenario], :observation, :value)
dropmissing!(wide)
X = Matrix(wide[:, vars])
Σ_y = cov(X)


# obs_list = []

# compute grad_Q: for each parameter and variable in observation set compute 1 / (high_0K - low_0K) and then 
# put in the \nabla_Q matrix

# df_wide = unstack(df_clean, [:param, :site, :observation], :scenario, :value)
df_wide.grad_0K = (df_wide.high_0K .- df_wide.low_0K)
df_wide.grad_4K = (df_wide.high_4K .- df_wide.low_4K)

Q_grad_0K = df_wide[:, [:param, :site, :observation, :grad_0K]]
dropmissing!(Q_grad_0K, :grad_0K)
Q_grad_0K = @subset(Q_grad_0K, isfinite.(:grad_0K))

combine(groupby(Q_grad_0K, :param), :grad_0K => mean => :mean_grad_0K, :grad_0K => std => :std_grad_0K)

# select observations in vars and average over sites and then pivot wider 
# Q_grad_0K = @subset(Q_grad_0K, :observation .∈ Ref(vars))
Q_grad_0K = combine(groupby(Q_grad_0K, [:param, :observation]), :grad_0K => mean => :grad_0K)
Q_grad_0K_wide = unstack(Q_grad_0K, [:param], :observation, :grad_0K)
order_indices_0K = [findfirst(==(p), Q_grad_0K_wide.param) for p in params_ordered if p in Q_grad_0K_wide.param]
Q_mat_0K = Matrix(Q_grad_0K_wide[order_indices_0K, vars])
Σ_θ_post_0K = inv(Q_mat_0K * cholesky_solve(Σ_y, Q_mat_0K') + inv(Σ₀))

mean_gs' * Σ_θ_post_0K * mean_gs





# use the old grad G matrix for the calculations since it is more robust 
@load "../ensemble_parameter_perturbations/data/bootstrap_sites/no_bootstrap/results.jld2" full_ig ∇G Σ_y Σ_0 constrained_params param_ordering
# make sure params_ordered is the same as param_ordering
# make sure when we subset information gain we are using the right variable ordering for each experiment. 
# make sure param ordering is the same as 

all_variables = get_all_variables(config)
sub_vars = create_var_list([], ["pr"], config["z_levels"]["shallow"])


function prod(sub_vars, all_vars, Σ_y, ∇G)
    bool_vec = in.(all_vars, Ref(Set(sub_vars)))

    Σ_y_sub = Σ_y[bool_vec, bool_vec]
    ∇G_sub = ∇G[bool_vec, :]
    # Diagonal(1 ./ diag(Σ_y_sub)) - if we just want to rescale obs

    return ∇G_sub' * cholesky_solve(Σ_y_sub, ∇G_sub)
end

prod(sub_vars, all_variables, Σ_y, ∇G)

mean_gs' *inv(prod(sub_vars, all_variables, Σ_y, ∇G) + inv(Σ₀)) * mean_gs

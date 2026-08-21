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


raw_data = CSV.read("dataframes/spp1_cfsites_intersect2.csv", DataFrame)

grad_df = g(params_ordered)
sites = unique(grad_df.site)

# filter to low latitudes for now only
t = transform(grad_df, :site => ByRow(x -> split(x, "_")) => [:lat, :lon, :site_date])
transform!(t, [:lat, :lon, :site_date] => ByRow((lat, lon, site_date) -> (parse(Float64, lat), parse(Float64, lon), convert(String, site_date))) => [:lat, :lon, :site_date])
filter!(row -> (abs(row.lat) <= 30.0), t)


df = CSV.read("grad_q_data.csv", DataFrame)
df = df[abs.(parse.(Float64, convert.(String, first.(split.(df.site, "_"))))) .<= 30.0, :]

config = YAML.load_file("experiment_config.yml");
vars = create_var_list(["ta"], [], config["z_levels"]["shallow"])

df_clean = @subset(df, isfinite.(:low_0K) .& isfinite.(:high_0K) .& isfinite.(:low_4K) .& isfinite.(:high_4K))
df_clean = stack(df, [:low_0K, :high_0K, :low_4K, :high_4K], variable_name = :scenario, value_name = :value)
df_clean = @subset(df_clean, isfinite.(:value) .& (:observation .∈ Ref(vars)))


rd_cl = transform(raw_data, :site => ByRow(x -> split(x, "_")) => [:lat, :lon, :site_date])
transform!(rd_cl, [:lat, :lon, :site_date] => ByRow((lat, lon, site_date) -> (parse(Float64, lat), parse(Float64, lon), convert(String, site_date))) => [:lat, :lon, :site_date])
filter!(row -> (abs(row.lat) <= 30.0), rd_cl)
mean(@subset(rd_cl, (:scenario .== "0K") .& .!isnan.(:cre)).cre)
mean(@subset(rd_cl, (:scenario .== "4K") .& .!isnan.(:cre)).cre)

gs_vectors = []
for site in sites
    grad_site = gs(site, params_ordered)
    push!(gs_vectors, grad_site)
end

var_reduction_100 = get_variance_reduction(
    config["var_names_prof"],
    config["var_names_int"],
    [100, 100, 4000],
    Σ₀,
    df,
    params_ordered,
    gs_vectors
)

prior_vars = @subset(var_reduction_100, :variable .== "ta").prior_var

# combine(groupby(var_reduction_100, :variable), :post_var => mean)

# df_sqrt = DataFramesMeta.select(var_reduction_100, :variable,
#                  :post_var => ByRow(sqrt) => :cre_uncertainty)

df_prior = DataFrame(
    variable = "prior", 
    cre_uncertainty = sqrt.(@subset(var_reduction_100, :variable .== "ta").prior_var)
)

dcre_uncertainty_reduction = vcat(df_sqrt, df_prior)

df_mean = combine(groupby(dcre_uncertainty_reduction, :variable),
               :cre_uncertainty => mean => :mean_cre_uncertainty)

sort!(df_mean, :mean_cre_uncertainty, rev = false)

targets = ["prior", "ta", "arup", "tke", "cl", "hus", "clw", "entr", "wa", "lwp", "pr"]

df_subset = @subset(df_mean, :variable .∈ Ref(targets))

using StatsPlots

barcolors = [v == "prior" ? :gray60 : :steelblue
             for v in df_subset.variable]

@df df_subset bar(
    :variable,
    :mean_cre_uncertainty / 4,
    ylabel = "ΔCRE uncertainty (W m⁻² K⁻¹)",
    xlabel = "Observed Variable",
    legend = false,
    grid = false,
    dpi = 300,
    fillcolor = barcolors,
    linecolor = :black,
    linewidth = 2,
    alpha = 0.9,
    size = (800, 450),
    xticks = :all, #(1:20, df_mean.variable),
    xtickfontsize = 10, 
    ytickfontsize = 10,
    legend_font_pointsize = 11,
    yguidefontsize = 12,
    xguidefontsize = 12,
    left_margin = 2mm,
)
savefig("figures/grads/cre_uncertainty_100m.png")


# pretty_names = Dict(
#     "ta"    => "Temperature",
#     "hus"   => "Specific hum",
#     "tke"   => "Turbulent KE",
#     "clw"   => "Cloud liquid water",
#     "prior" => "Prior",
#     "cli"   => "Cloud ice",
#     "hur"   => "Relative humidity",
#     "cl"    => "Cloud fraction",
#     "clivi" => "Cloud ice water content",
#     "lwp"   => "Liquid water path",
#     "pr"    => "Precipitation",
#     "rsut"  => "TOA up SW",
#     "rsutcs"=> "TOA up SW clear-sky",
#     "rlut"  => "TOA up LW",
#     "rlutcs"=> "TOA up LW clear-sky",
#     "dsevi" => "Dry static energy",
# )

# df_plot = transform(
#     df_mean,
#     :variable => ByRow(v -> get(pretty_names, v, v)) => :variable_pretty
# )





@df dcre_uncertainty_reduction StatsPlots.boxplot(
    :variable, 
    :cre_uncertainty / 4,
    ylabel = "CRE Uncertainty (W/m²)",
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
savefig("figures/grads/cre_box_uncertainty_100m.png")
# compute the gradient with respect to CRE for each parameter perturbation

import YAML
import JLD2
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis
using DataFrames
using CSV
using NaNStatistics

# want a vector with d(CRE)/d(param_i) for each parameter if
# we have CRE(param_i + delta) and CRE(param_i - delta)

# first need to compute the CRE for each simulation

config = YAML.load_file("experiment_config.yml")
# output_dir = "spp1_cfsites" #config["output_dir"]



# parameters = basename.(glob(joinpath(output_dir, "*")))
# perturbations = collect(Set(basename.(glob(joinpath(output_dir, "*/*")))))
# sites = collect(Set(basename.(glob(joinpath(output_dir, "*/*/*01")))))
# all_paths = joinpath.(vec(collect(Iterators.product([parameters, perturbations, sites]...))))
# sim_dirs = joinpath.(output_dir, all_paths, "output_active")

function get_metadata(output_dir)
    parameters = basename.(glob(joinpath(output_dir, "*")))
    perturbations = collect(Set(basename.(glob(joinpath(output_dir, "*/*")))))
    sites = collect(Set(basename.(glob(joinpath(output_dir, "*/*/*01")))))
    all_paths = joinpath.(vec(collect(Iterators.product([parameters, perturbations, sites]...))))
    return parameters, perturbations, sites, all_paths
end

parameters0K, perturbations0K, sites0K, all_paths0K = get_metadata("spp1_cfsites")
parameters4K, perturbations4K, sites4K, all_paths4K = get_metadata("spp1_cfsites_p4K")

# get indices of matching sites 
# match_0K = findall(in(sites4K), sites0K)
# match_4K = findall(in(sites0K), sites4K)
# get intersection of sites 
common_sites = intersect(sites0K, sites4K)

# filter paths by match 
paths0K_filtered = filter(path -> any(occursin.(common_sites, path)) & any(occursin.(["/0.25/", "/0.75/"], path)), all_paths0K)
paths4K_filtered = filter(path -> any(occursin.(common_sites, path)) & any(occursin.(["/0.25/", "/0.75/"], path)), all_paths4K)

paths0K_simdirs = joinpath.("spp1_cfsites", paths0K_filtered, "output_active")
paths4K_simdirs = joinpath.("spp1_cfsites_p4K", paths4K_filtered, "output_active")

function compute_cre(path, config = config)
    try
        sim_dir = SimDir(path)

        SW_all = ClimaAnalysis.get(sim_dir; short_name = "rsut")
        SW_cs = ClimaAnalysis.get(sim_dir; short_name = "rsutcs")
        LW_all = ClimaAnalysis.get(sim_dir; short_name = "rlut")
        LW_cs = ClimaAnalysis.get(sim_dir; short_name = "rlutcs")

        # compute cre
        CRE = (SW_all - SW_cs) + (LW_all - LW_cs)

        if CRE.dims["time"][end] < config["reduction_end_time"]
            return NaN
        end

        # average in time over spun up period

        CRE_slice = window(CRE, "time"; left = config["reduction_start_time"], right=config["reduction_end_time"])
        cre_time_avg = average_time(slice(CRE_slice, x=0, y=0))

        return cre_time_avg.data[1]
    catch e 
        println("Error computing CRE for path $path: $e")
        if e isa InterruptException
            throw(e)
        end
        return NaN
    end
end

CRE_p0K = [compute_cre(path) for path in paths0K_simdirs]
# CRE_p4K = [compute_cre(path) for path in paths4K_simdirs]
@elapsed CRE_p4K = compute_cre.(paths4K_simdirs)

# get the simulation information from the string, and build a dataframe
df = DataFrame(param = String[], perturb = Float64[], site = String[], scenario = String[], cre = Float64[])

for (i, path) in enumerate(paths0K_filtered)
    param_name, perturb, site = String.(split(path, "/"))
    perturb = parse(Float64, perturb)
    cre = CRE_p0K[i]
    push!(df, (param_name, perturb, site, "0K", cre))
end
for (i, path) in enumerate(paths4K_filtered)
    param_name, perturb, site = String.(split(path, "/"))
    perturb = parse(Float64, perturb)
    cre = CRE_p4K[i]
    push!(df, (param_name, perturb, site, "4K", cre))
end

CSV.write("dataframes/spp1_cfsites_intersect2.csv", df)

df = CSV.read("dataframes/spp1_cfsites_intersect2.csv", DataFrame)


# params = unique(df.param)

# val_df = filter(rows -> (rows.site == site) & (rows.param == param), df)
# cre_param_low_scen_0k = filter(row -> (row.perturb == 0.25) & (row.scenario == "0K"), val_df).cre[1]
# cre_param_high_scen_0k = filter(row -> (row.perturb == 0.75) & (row.scenario == "0K"), val_df).cre[1]
# cre_param_low_scen_4k = filter(row -> (row.perturb == 0.25) & (row.scenario == "4K"), val_df).cre[1]
# cre_param_high_scen_4k = filter(row -> (row.perturb == 0.75) & (row.scenario == "4K"), val_df).cre[1]

# ΔCRE_param_low = cre_param_low_scen_4k - cre_param_low_scen_0k
# ΔCRE_param_high = cre_param_high_scen_4k - cre_param_high_scen_0k

# dΔCRE_dparam = (ΔCRE_param_high - ΔCRE_param_low)


function get_dcre_dparam(df, site, param)
    sub_df = filter(rows -> (rows.site == site) & (rows.param == param), df)
    cre_param_low_scen_0k = filter(row -> (row.perturb == 0.25) & (row.scenario == "0K"), sub_df).cre[1]
    cre_param_high_scen_0k = filter(row -> (row.perturb == 0.75) & (row.scenario == "0K"), sub_df).cre[1]
    cre_param_low_scen_4k = filter(row -> (row.perturb == 0.25) & (row.scenario == "4K"), sub_df).cre[1]
    cre_param_high_scen_4k = filter(row -> (row.perturb == 0.75) & (row.scenario == "4K"), sub_df).cre[1]

    ΔCRE_param_low = cre_param_low_scen_4k - cre_param_low_scen_0k
    ΔCRE_param_high = cre_param_high_scen_4k - cre_param_high_scen_0k

    dΔCRE_dparam = (ΔCRE_param_high - ΔCRE_param_low)
    return dΔCRE_dparam
end

dcre_df = DataFrame(param = String[], site = String[], dcre_dparam = Float64[])
parameters = unique(df.param)
for param in parameters
    for site in common_sites
        dcre = get_dcre_dparam(df, site, param)
        push!(dcre_df, (param, site, dcre))
    end
end
filter!(:dcre_dparam => isfinite, dcre_df)

# focus on only tropical low clouds 
t = transform(dcre_df, :site => ByRow(x -> split(x, "_")) => [:lat, :lon, :site_date])
transform!(t, [:lat, :lon, :site_date] => ByRow((lat, lon, site_date) -> (parse(Float64, lat), parse(Float64, lon), convert(String, site_date))) => [:lat, :lon, :site_date])

filter!(row -> (abs(row.lat) <= 30.0), t)

# see the mean effect for each param 
by_param2 = combine(groupby(t, :param)) do sdf
    v = sdf.dcre_dparam
    (; mean = mean(v), q10 = quantile(v, 0.10), q90 = quantile(v, 0.90))
end



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
# plot 

groups = groupby(t, :param, sort = false)
params = [g.param[1] for g in groups]
data = [g.dcre_dparam[:] for g in groups]
labels = [get(short_name_map, p, p) for p in params]

using StatsPlots
using StatsBase

plt = StatsPlots.boxplot(data;
    #label = labels,         # legend labels
    xticks = (1:length(labels), labels),  # x-axis tick positions + labels
    xlabel = "Parameter",
    ylabel = "ΔCRE over Parameter IQR (W/m²)",
    title = "Distribution by Parameter",
    legend = false,
    linewidth = 2,
    color = :royalblue,
    mediancolor = :white,
    whisker_width = 0.5,
    outliercolor = :red,
    ylims = (-5, 5)
)

# Save to file (png/pdf/svg etc.)
savefig(plt, "figures/grads/boxplot.png")

df_samples = []
for i in 1:1000
    site_sample = StatsBase.sample(unique(t.site), length(unique(t.site)); replace = true)
    sub_df = filter(row -> row.site ∈ site_sample, t)
    sub_df = combine(groupby(sub_df, :param, sort = false), :dcre_dparam => mean => :mean_dcre_dparam)
    sub_df[!, :replicate] .= i 
    push!(df_samples, sub_df)
end

df_all = vcat(df_samples...) # combine 

# df_all.param = CategoricalArray(df_all.param; levels = params, ordered = true)


# @df df_all StatsPlots.boxplot(
#     :param, :mean_dcre_dparam;
#     ylabel = "Mean ΔCRE / Δparam",
#     xlabel = "Parameter",
#     xticks = (1:length(labels), labels),  # x-axis tick positions + labels
#     title = "Bootstrap distribution of parameter sensitivities",
#     legend = false,
#     color = :royalblue,
#     mediancolor = :white,
#     whisker_width = 0.5
# )
# savefig("figures/grads/bootstrap_boxplot.png")

# ci = combine(groupby(df_all, :param)) do sdf
#            q = quantile(sdf.mean_dcre_dparam, [0.025, 0.5, 0.975])
#            (; param = first(sdf.param), lo = q[1], med = q[2], hi = q[3])
# end

summaries = combine(groupby(df_all, :param)) do sdf
    μ = mean(sdf.mean_dcre_dparam)
    q90 = quantile(sdf.mean_dcre_dparam, [0.05, 0.95])
    q95 = quantile(sdf.mean_dcre_dparam, [0.025, 0.975])
    sig_level = if !(q95[1] <= 0 <= q95[2])
        "95% significant"
    elseif !(q90[1] <= 0 <= q90[2])
        "90% significant"
    else
        "insignificant"
    end
    (; param = first(sdf.param), mean = μ, sig_level)
end


# --- 2. Order parameters by mean effect size
ordered_params = sort(summaries, :mean).param
sorted_labels = [get(short_name_map, p, p) for p in ordered_params]

# --- 3. Map to categorical variable for safe ordering
df_all.param = CategoricalArray(df_all.param;
    levels = ordered_params, ordered = true)

# --- 4. Define color map (light if not influential)
palette = Dict(
    "95% significant" => :royalblue,
    "90% significant" => :cornflowerblue,
    "insignificant"   => :lightgray
)

df_all.sig_level = [summaries[summaries.param .== p, :sig_level][1] for p in df_all.param]
df_all.color = [palette[s] for s in df_all.sig_level]

# --- 5. Plot
@df df_all StatsPlots.boxplot(
    :param, :mean_dcre_dparam;
    group = :param,             # ensures each param grouped separately
    color = df_all.color,
    xticks = (1:length(sorted_labels), sorted_labels),
    mediancolor = :white,
    whisker_width = 0.5,
    legend = :topleft,
    xlabel = "Parameter",
    ylabel = "Mean ΔCRE / Δparam (W/m²)",
    title = "Sensitivity of Tropical +4K ΔCRE to Param Perturbations",
    dpi = 300,
    label = "",
)

for (label, col) in palette
    StatsPlots.plot!([NaN], [NaN];
        seriestype = :scatter,
        markerstrokecolor = :transparent,
        markercolor = col,
        label = label,
    )
end

savefig("figures/grads/bootstrap_boxplot_CI.png")




# we'll again use fixed effects regression to compute the gradients by fixing over the sites 
using FixedEffectModels
gradients = DataFrame(param = String[], scenario = String[], gradient = Float64[])
for param in parameters
    # compute for 0K scenario
    sub_df = filter(row -> (row.param == param) & !isnan(row.cre) & (row.perturb ∈ [0.3, 0.7]) & (row.scenario == "0K"), df)
    model = reg(sub_df, @formula(cre ~ perturb + fe(site)))
    gradient = coef(model)[1]  # coefficient for perturb
    push!(gradients, (param, "0K", gradient))
    # compute for 4K scenario
    sub_df = filter(row -> (row.param == param) & !isnan(row.cre) & (row.scenario == "4K"), df)
    model = reg(sub_df, @formula(cre ~ perturb + fe(site)))
    gradient = coef(model)[1]  # coefficient for perturb
    push!(gradients, (param, "4K", gradient))
end
CSV.write("dataframes/gradients_cre_p0k.csv", gradients)
println("Completed computing gradients of CRE with respect to parameters")


# visualize the gradient differences 
# grad_p0K = rename(CSV.read("dataframes/gradients_cre_p0k.csv", DataFrame), :gradient => :grad_p0K)
# grad_p4K = rename(CSV.read("dataframes/spp4pk_grad.csv", DataFrame), :gradient => :grad_p4K)
# # join dataframes 
# grad_combined = innerjoin(grad_p0K, grad_p4K, on = :param)
# grad_stacked = DataFrames.stack(grad_combined, [:grad_p0K, :grad_p4K], variable_name = :simulation, value_name = :gradient)


fig = Figure(size = (800, 600))

params = unique(gradients.param)
scenarios = ["0K", "4K"]
colors = Makie.wong_colors()

ax = Axis(fig[1, 1]; 
        xlabel = "Parameter", 
        ylabel = "CRE Gradient (W/m² per Parameter IWR range)",
        xticks = (1:length(params), [short_name_map[param] for param in params]),
        xticklabelrotation = π/4,
        title = "CRE Parameter Gradients in current and +4K warming scenarios",
)

barplot!(ax, map(s -> findfirst(isequal(s), params), gradients.param), grad_stacked.gradient,
    dodge = Int.(map(s -> findfirst(isequal(s), scenarios), gradients.scenario)),
    color = colors[Int.(map(s -> findfirst(isequal(s), scenarios), gradients.scenario))],
)

elements = [PolyElement(polycolor = colors[i]) for i in 1:length(scenarios)]
title = "Scenario"
Legend(fig[1,1], elements, scenarios, title;
    tellheight = false,
    tellwidth = false,
    margin = (10, 10, 10, 10),
    halign = :center, valign = :top,
)


save("figures/grads/cre_gradient_comparison_p0k_p4k.png", fig)

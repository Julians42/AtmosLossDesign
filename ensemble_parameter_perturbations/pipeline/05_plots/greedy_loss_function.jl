# Greedily add variables (fixed at 500m resolution), one at a time, by
# whichever gives the largest marginal information gain. Repeated across the
# 100 bootstrap replicates to quantify selection-order uncertainty.
#
# `data/cache/greedy_loss_function*.csv` are resumable caches: the bootstrap
# loop below reads a pre-existing CSV and only continues from `boot_i = 72`
# onward because that's where a prior run left off. To recompute from
# scratch, delete `data/cache/greedy_loss_function_bootstrap.csv` and change
# the range below to `1:100`.

import EnsembleKalmanProcesses as EKP
import YAML
import ClimaCalibrate as CAL
using Statistics
using DataFrames
using CSV
using CairoMakie
using LinearAlgebra
using FixedEffectModels
import TOML
using Glob
using JLD2

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "methods.jl"))
include(joinpath(PROJECT_ROOT, "src", "parameter_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiment.jl"))

cache_dir = joinpath(PROJECT_ROOT, "data", "cache")
mkpath(cache_dir)

config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
resolve_config_paths!(config, PROJECT_ROOT)
all_variables = get_all_variables(config, Config_cfsites_deep())

@load joinpath(PROJECT_ROOT, "data", "bootstrap_sites", "no_bootstrap", "results.jld2") full_ig ∇G Σ_y Σ_0 constrained_params param_ordering

all_var_roots = vcat(config["var_names_int"], config["var_names_prof"])
remaining_var_roots = copy(all_var_roots)
best_ordering = []
best_igs = []

basic_config = Dict(
        "var_names_int" => [],
        "var_names_prof" => [],
        "z_levels" => Dict("shallow" => [100, 100, 4000],
                           "deep" => [100, 100, 4000]),
        "exp_name" => ""
)

greedy_loss_function_csv = joinpath(cache_dir, "greedy_loss_function.csv")
greedy_loss_function_bootstrap_csv = joinpath(cache_dir, "greedy_loss_function_bootstrap.csv")

ig_df = DataFrame(iter=Int[], var=String[], ig=Float64[])
ordered_best_list = []
iter = 0
while length(remaining_var_roots) > 0
    ig_iter_value = Dict()
    for (i, var_root) in enumerate(remaining_var_roots)
        new_config = deepcopy(basic_config)

        if var_root in config["var_names_int"]
            push!(new_config["var_names_int"], var_root)
        else
            push!(new_config["var_names_prof"], var_root)
        end
        for var in ordered_best_list
            if var in config["var_names_int"]
                push!(new_config["var_names_int"], var)
            else
                push!(new_config["var_names_prof"], var)
            end
        end

        var_levels = get_all_variables(new_config, Config_cfsites_deep())
        sub_ig = subset_info_gain(var_levels, all_variables, Σ_y, Σ_0, ∇G)
        println(var_root, ": ", sub_ig)
        ig_iter_value[var_root] = sub_ig
    end
    best_new_var = findmax(ig_iter_value)[2]
    best_ig = findmax(ig_iter_value)[1]
    push!(ordered_best_list, best_new_var)
    println("Best new variable is $best_new_var with ig: $(best_ig)")
    filter!(x -> x != best_new_var, remaining_var_roots)
    iter += 1
    push!(ig_df, (iter=iter, var=best_new_var, ig=best_ig))
    CSV.write(greedy_loss_function_csv, ig_df)
end

# Plot information gain by iteration with annotations of added variable
fig = Figure(size = (1000, 500))
ax = Axis(fig[1, 1];
                   title = "Greedy variable selection: information gain by iteration",
                   xticklabelrotation = 0,
                   ylabel = "Information Gain",
                   xlabel = "Iteration",
                   xlabelsize = 16, ylabelsize = 16,
                   xticklabelsize = 11, yticklabelsize = 11)
xs = 1:nrow(ig_df)
barplot!(ax, xs, ig_df.ig; color = :steelblue, alpha = 1., strokecolor = :black, strokewidth = 2)
ax.xticks = xs

# annotate variable names at the top of each bar
ypad = 0.02 * maximum(ig_df.ig)
for (i, row) in enumerate(eachrow(ig_df))
    text!(ax, row.var; position = (i, row.ig + ypad), align = (:center, :bottom), fontsize = 10, color = :black)
end

save(joinpath(PROJECT_ROOT, "plots", "greedy_loss_function.png"), fig)

################################################################################
# repeat for the bootstrap sites

basic_config = Dict(
        "var_names_int" => [],
        "var_names_prof" => [],
        "z_levels" => Dict("shallow" => [100, 100, 4000],
                           "deep" => [100, 100, 4000]),
        "exp_name" => ""
)

ig_df_bootstrap = isfile(greedy_loss_function_bootstrap_csv) ? CSV.read(greedy_loss_function_bootstrap_csv, DataFrame) : DataFrame(iter=Int[], var=String[], ig=Float64[], bootstrap=Int[])
for boot_i in 72:100
    @load joinpath(PROJECT_ROOT, "data", "bootstrap_sites", "bootstrap_$boot_i", "results.jld2") full_ig ∇G Σ_y Σ_0 constrained_params param_ordering

    remaining_var_roots = copy(all_var_roots)
    ordered_best_list = []
    iter = 0
    while length(remaining_var_roots) > 0
        ig_iter_value = Dict()
        for (i, var_root) in enumerate(remaining_var_roots)
            new_config = deepcopy(basic_config)

            if var_root in config["var_names_int"]
                push!(new_config["var_names_int"], var_root)
            else
                push!(new_config["var_names_prof"], var_root)
            end
            for var in ordered_best_list
                if var in config["var_names_int"]
                    push!(new_config["var_names_int"], var)
                else
                    push!(new_config["var_names_prof"], var)
                end
            end

            var_levels = get_all_variables(new_config, Config_cfsites_deep())
            sub_ig = subset_info_gain(var_levels, all_variables, Σ_y, Σ_0, ∇G)
            println(var_root, ": ", sub_ig)
            ig_iter_value[var_root] = sub_ig
        end
        best_new_var = findmax(ig_iter_value)[2]
        best_ig = findmax(ig_iter_value)[1]
        push!(ordered_best_list, best_new_var)
        println("Best new variable is $best_new_var with ig: $(best_ig)")
        filter!(x -> x != best_new_var, remaining_var_roots)
        iter += 1
        push!(ig_df_bootstrap, (iter=iter, var=best_new_var, ig=best_ig, bootstrap = boot_i))
        CSV.write(greedy_loss_function_bootstrap_csv, ig_df_bootstrap)
    end
end

################################################################################

ig_df_bootstrap = CSV.read(greedy_loss_function_bootstrap_csv, DataFrame)
ig_df = CSV.read(greedy_loss_function_csv, DataFrame)

# Plot information gain by iteration with annotations of added variable
fig = Figure(size = (1200, 500))
ax = Axis(fig[1, 1];
                   title = "Greedy variable selection: information gain by iteration",
                   xticklabelrotation = 0,
                   ylabel = "Information Gain",
                   xlabel = "Iteration",
                   xlabelsize = 16, ylabelsize = 16,
                   xticklabelsize = 11, yticklabelsize = 11)
xs = 1:nrow(ig_df)
barplot!(ax, xs, ig_df.ig; color = :steelblue, alpha = 1., strokecolor = :black, strokewidth = 2)
ax.xticks = xs

# annotate variable names at the top of each bar
ypad = 0.02 * maximum(ig_df.ig)
for (i, row) in enumerate(eachrow(ig_df))
    text!(ax, row.var; position = (i, row.ig + ypad), align = (:center, :bottom), fontsize = 10, color = :black)
end

counts = combine(groupby(ig_df_bootstrap, [:var, :iter]), nrow => :count)

pivoted = unstack(counts, :iter, :var, :count)

vars  = names(pivoted)[2:end]
mat   = Matrix(pivoted[:, vars]) ./ 100
iters = pivoted.iter

ax2 = Axis(fig[1,2];
          xlabel="Variable",
          ylabel="Greedy Selection Iteration",
          title="Uncertainty Quantification of Greedy Selection with Site-level Bootstrap",
          xticks=(1:length(vars), vars),
          yticks=(1:length(iters), string.(iters)),
          xticklabelrotation=π/2)

hm = heatmap!(ax2, 1:length(vars), 1:length(iters), mat; colormap=Reverse(:lapaz))
Colorbar(fig[1, 3], hm, label = "Frequency")

save(joinpath(PROJECT_ROOT, "plots", "greedy_loss_function_bootstrap.png"), fig)

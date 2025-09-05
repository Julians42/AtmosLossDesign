# fix a resolution
# add variables greedily by measuring the information gain 

using DataFrames
using CSV
using CairoMakie

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
    # save dataframe
    CSV.write("greedy_loss_function.csv", ig_df)
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

save("plots/greedy_loss_function.png", fig)


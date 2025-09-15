using DataFrames
using CairoMakie

resolutions = [100, 200, 300, 400, 500, 800, 1000, 1300, 2000]

"""
    filter_variables_to_resolution(variables, resolution)

Return only those variables that encode the given resolution as a suffix after an underscore, e.g. "ta_100".
Variables without an underscore are ignored.
"""
function filter_variables_to_resolution(variables, resolution; variable_filter = "")
    filt_vars = String[]
    for var in variables
        if occursin('_', var)
            parts = split(var, "_")
            res = tryparse(Int, parts[end])            
            if res !== nothing && res % resolution == 0 && "$(parts[1])" == variable_filter
                push!(filt_vars, var)
            end
        end
    end
    return filt_vars
end

# Compute the marginal information gain for the variables at each resolution
vars_of_interest = ["entr", "arup", "clw", "tke", "hur", "ta", "cl", "hus", "cli", "wa"]
ig_df = DataFrame(var=String[], resolution=Int[], ig=Float64[])

for var in vars_of_interest
    for res in resolutions
        variables = filter_variables_to_resolution(all_variables, res, variable_filter = var)
        if isempty(variables)
            continue
        end
        ig = subset_info_gain(variables, all_variables, Σ_y, Σ_0, ∇G)
        push!(ig_df, (var=var, resolution=res, ig=ig))
    end
end

ig_df.n_points = floor.(4000 ./ ig_df.resolution)

# Plot: resolution on x-axis, IG on y-axis, one line per variable with legend
fig = Figure(size = (900, 500))
ax = Axis(fig[1, 1]; xlabel = "Observations below 4000m", ylabel = "Information Gain", title = "Measurement resolution effect on calibratable information",
          xlabelsize = 14, ylabelsize = 14, xticklabelsize = 12, yticklabelsize = 12)

# for var in vars_of_interest
#     sub = ig_df[ig_df.var .== var, :]
#     if nrow(sub) == 0
#         continue
#     end
#     sub_sorted = sort(sub, :n_points)
#     lines!(ax, sub_sorted.n_points, sub_sorted.ig; label = var)
# end
palette = ColorSchemes.tab10
for (i, var) in enumerate(vars_of_interest)
    sub = ig_df[ig_df.var .== var, :]
    if nrow(sub) == 0
        continue
    end
    sub_sorted = sort(sub, :n_points)
    lines!(ax, sub_sorted.n_points, sub_sorted.ig;
        label = var,
        color = palette[i],
        linewidth = 3, # bolder
    )
end

axislegend(ax, position = :lt)
save("plots/resolution_information_gain.png", fig)



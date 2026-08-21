# Legacy regression-coefficient / SVD-based analysis, superseded by the
# information-gain approach in ../src/methods.jl + ../src/experiment.jl. Kept
# for reference; not part of the actively maintained pipeline (see
# ../README.md and README.md in this directory).
#
# Expects the including script to have already done:
#   using DataFrames, CairoMakie, LinearAlgebra, LaTeXStrings, FixedEffectModels
# and to have already included ../src/parameter_io.jl (for normalize_parameters)
# and ../src/methods.jl (for normalize_statistics).

_legacy_plots_dir() = normpath(joinpath(@__DIR__, "..", "plots"))

function create_parameter_dataframe(rootdir, param_stats_df; num_members=100)
    all_vals = []
    all_keys = []
    members = []

    for i in 1:num_members
        member = "member_" * lpad(i, 3, "0")
        push!(members, member)

        norm_params = normalize_parameters("$rootdir/$member/parameter.toml", param_stats_df)

        val_array = []
        key_array = []
        for (key, value) in norm_params
            if value isa Array
                for (i, v) in enumerate(value)
                    push!(val_array, v)
                    push!(key_array, key * "_" * string(i))
                end
            else
                push!(val_array, value)
                push!(key_array, key)
            end
        end

        push!(all_vals, val_array)
        push!(all_keys, key_array)
    end

    parameter_df = DataFrame(member = members)
    parameter_df[!, :values] = all_vals
    parameter_df[!, :keys] = all_keys

    return parameter_df
end

function postprocess_dataframe(df::DataFrame, param_stats_df::DataFrame, num_members::Int, rootdir::String)
    df = normalize_statistics(df)
    parameter_df = create_parameter_dataframe(rootdir, param_stats_df, num_members = num_members)
    return innerjoin(df, parameter_df; on=:member)
end

function compute_regression_coefficients_optimized(df_cleaned::DataFrame, informing_variables)
    @info "Grouping dataframe by variable..."
    grouped_df = groupby(df_cleaned, :variable)

    first_var_data = grouped_df[1]
    param_cols = Symbol.(first_var_data.keys[1])
    param_terms = join(string.(param_cols), " + ")
    formula_str = "normalized_statistic ~ $param_terms + fe(site)"
    test_formula = eval(Meta.parse("@formula($formula_str)"))

    coef_ar = []
    coef_names = []
    failed_vars = []

    @info "Processing $(length(informing_variables)) variables..."

    for (i, var) in enumerate(informing_variables)
        try
            if haskey(grouped_df, (variable = var,))
                var_data = grouped_df[(variable = var,)]

                n_rows = nrow(var_data)
                n_params = length(param_cols)
                param_matrix = Matrix{Float64}(undef, n_rows, n_params)

                for (row_idx, values) in enumerate(var_data.values)
                    param_matrix[row_idx, :] = values
                end

                parameter_df = DataFrame(param_matrix, param_cols)
                combined_df = hcat(
                    select(var_data, [:member, :site, :normalized_statistic]),
                    parameter_df
                )

                model = reg(combined_df, test_formula)
                push!(coef_ar, model.coef)
                push!(coef_names, model.coefnames)
            else
                println("Variable $var not found in data")
                push!(failed_vars, i)
            end
        catch e
            if e isa InterruptException
                throw(e)
            else
                println("$var failed: $e")
                push!(failed_vars, i)
            end
        end

        if i % 50 == 0
            @info "Processed $i/$(length(informing_variables)) variables"
        end
    end

    coef_ar = hcat(coef_ar...)
    return coef_ar, coef_names[1], failed_vars
end

function plot_regression_analysis(B_matrix::Matrix, B_names::Vector, param_names::Vector, config::Dict)
    fig1 = Figure(size = (500, 400))
    ax1 = Axis(fig1[1,1],
        xlabel = "Principle Component Dimension",
        xlabelsize = 12,
        ylabel = L"\text{Eigenvalue of } H_\theta H_\theta^T",
        title = "Learnable Dimension")
    scatter!(ax1, 1:length(svd(B_matrix * transpose(B_matrix)).S),
             svd(B_matrix * transpose(B_matrix)).S)
    save(joinpath(_legacy_plots_dir(), "$(config["exp_name"])_scree_param_informed.png"), fig1)

    parameter_informed_diag = diag(B_matrix * transpose(B_matrix))
    variable_informed_diag = diag(transpose(B_matrix) * B_matrix)

    fig3 = Figure(size = (400, 400))
    ax3 = Axis(fig3[1,1],
        ylabelsize = 15,
        xlabelsize = 15,
        titlesize = 15,
        yticklabelsize = 12,
        xticklabelsize = 10,
        xlabel = "Parameter",
        ylabel = "Normalized Identifiability",
        title = "Parameter Identifiability for Diagnostic SCM")

    sorted_param_inds = sortperm(parameter_informed_diag, rev=true)
    sorted_param_vals = parameter_informed_diag[sorted_param_inds]
    sorted_param_names = param_names[sorted_param_inds]

    barplot!(ax3, 1:length(sorted_param_vals), sorted_param_vals ./ norm(sorted_param_vals, 1))
    ax3.xticks = (1:length(sorted_param_names), sorted_param_names)
    ax3.xticklabelrotation = π/2
    save(joinpath(_legacy_plots_dir(), "$(config["exp_name"])_param_informed.png"), fig3)

    fig4 = Figure(size = (800, 400))
    ax4 = Axis(fig4[1,1],
        xlabel = "Variable",
        ylabel = "Diagonal value of BᵀB",
        title = "Variable informative dimension ($(config["exp_name"]))")

    sorted_var_inds = sortperm(variable_informed_diag, rev=true)
    sorted_var_vals = variable_informed_diag[sorted_var_inds]
    sorted_var_names = B_names[sorted_var_inds]

    important_var_inds = sorted_var_vals .> 0.05 * sorted_var_vals[1]
    sorted_var_vals = sorted_var_vals[important_var_inds]
    sorted_var_inds = sorted_var_inds[important_var_inds]
    sorted_var_names = sorted_var_names[important_var_inds]

    barplot!(ax4, 1:length(sorted_var_vals), sorted_var_vals)
    ax4.xticks = (1:length(sorted_var_names), sorted_var_names)
    ax4.xticklabelrotation = π/2
    save(joinpath(_legacy_plots_dir(), "$(config["exp_name"])_var_informed.png"), fig4)

    plot_variable_informing_analysis(B_matrix, B_names,
                                     config["var_names_prof"], config["var_names_int"], config["exp_name"])

    return sorted_var_vals, sorted_var_inds, sorted_var_names
end

function plot_variable_informing_analysis(reg_coefs::Matrix, informing_variables::Vector{String},
                                       var_names_prof::Vector{String}, var_names_int::Vector{String},
                                       exp_name::String)
    var_informing_ar = []

    for profile_var in var_names_prof
        matching_indx = getindex.(split.(informing_variables, "_"), 1) .== profile_var
        var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
        println(profile_var)
        push!(var_informing_ar, [profile_var, - 1/2 *log(abs(det(I - var_informed_ar))) / size(var_informed_ar, 1)])
    end

    for single_var in var_names_int
        try
            matching_indx = getindex.(split.(informing_variables, "_"), 1) .== single_var
            var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
            println(single_var, " ", var_informed_ar[1,1])
            println(single_var, " ", sum(matching_indx))
            push!(var_informing_ar, [single_var, -1/2 * log(1- var_informed_ar[1,1])])
        catch
            println(single_var, " failed")
        end
    end

    var_informing_matrix = hcat(var_informing_ar...)
    var_informing_df = DataFrame(
        var_name = var_informing_matrix[1, :],
        var_informing = var_informing_matrix[2, :]
    )

    sorted_df = sort(var_informing_df, :var_informing, rev=true)

    fig_var_informing = Figure(size = (800, 400))
    ax_var_informing = Axis(fig_var_informing[1,1],
        xlabel = "Observation",
        ylabel = "Normalized Observation Informativeness",
        title = "Observation Informativeness for Diagnostic SCM")

    barplot!(ax_var_informing, 1:length(sorted_df.var_informing), sorted_df.var_informing ./ norm(sorted_df.var_informing, 1))
    ax_var_informing.xticks = (1:length(sorted_df.var_name), sorted_df.var_name)
    ax_var_informing.xticklabelrotation = π/2
    save(joinpath(_legacy_plots_dir(), "$(exp_name)_var_informing_integrated.png"), fig_var_informing)
end

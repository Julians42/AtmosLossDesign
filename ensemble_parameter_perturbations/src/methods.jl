# Pure, in-memory statistics/math routines for the information-gain analysis.
#
# Nothing in this file does file I/O or assumes a particular directory layout
# or experiment_config.yml schema - every function takes arrays/DataFrames/
# matrices already in memory and returns a result. That makes this file safe
# to reuse against any similarly-shaped data (e.g. someone else's CSV of
# parameters and statistics), independent of how the data was produced.
#
# Dataset-specific loading/orchestration lives in experiment.jl; path-based
# file I/O lives in parameter_io.jl.
#
# Expects the including script to have already done:
#   using Statistics, LinearAlgebra, DataFrames, DataFramesMeta
#   using FixedEffectModels
#   import EnsembleKalmanProcesses as EKP

abstract type GradientApproximationMethod end
struct Linear <: GradientApproximationMethod end
struct GP <: GradientApproximationMethod end

function cholesky_solve(Σ_y, b)
    Σ_y_reg = Σ_y + 1e-10 * I
    if !issymmetric(Σ_y)
        Σ_y_reg = 0.5 * (Σ_y_reg + Σ_y_reg')
    end
    L = cholesky(Σ_y_reg).L
    return L' \ (L \ b)
end

"""
    information_gain(variables, df, constrained_params, param_ordering, grad_method)

Pure core of the information-gain calculation: given a dataframe of normalized
statistics and an already-loaded, already-normalized parameter ensemble
(`constrained_params`, `param_ordering`), compute the prior/posterior
information gain and the fitted Jacobian/covariances.

If you only have a `rootdir` of per-member `parameter.toml` files, load and
normalize them first with `load_member_parameters` + `normalize_constrained_parameters`
(see parameter_io.jl / methods.jl), or use the convenience wrapper
`information_gain(variables, df, grad_method; rootdir, prior_path)` in
experiment.jl.
"""
function information_gain(variables, df, constrained_params, param_ordering, grad_method::GradientApproximationMethod)
    df_sub = df[in.(df.variable, Ref(Set(variables))), :]
    Σ_0 = cov(constrained_params') # d x d matrix
    Σ_y = observational_covariance(df_sub, variables) # m x m matrix
    ∇G = gradient_approximation(df_sub, variables, constrained_params, param_ordering, grad_method)'

    # compute information gain relative to the prior
    ig = 1 / 2 * logdet(∇G' * cholesky_solve(Σ_y, ∇G) * Σ_0 + I)
    return ig, ∇G, Σ_y, Σ_0
end

function subset_info_gain(sub_vars, all_vars, Σ_y, Σ_0, ∇G)
    bool_vec = in.(all_vars, Ref(Set(sub_vars)))

    Σ_y_sub = Σ_y[bool_vec, bool_vec]
    ∇G_sub = ∇G[bool_vec, :]

    return det(∇G_sub' * cholesky_solve(Σ_y_sub, ∇G_sub) + inv(Σ_0)) / det(inv(Σ_0)) - 1
end

function subset_parameter_informedness(sub_vars, all_vars, Σ_y, Σ_0, ∇G)
    bool_vec = in.(all_vars, Ref(Set(sub_vars)))
    Σ_y_sub = Σ_y[bool_vec, bool_vec]
    ∇G_sub = ∇G[bool_vec, :]
    return diag(∇G_sub' * cholesky_solve(Σ_y_sub, ∇G_sub))
end

"""
    observational_covariance(df, vars)

Computes the observational covariance over all samples and sites. Expects
`df` to already have `:site`, `:variable`, `:member`, `:normalized_statistic`
columns (see `normalize_statistics`).
"""
function observational_covariance(df, vars)
    df_clean = @subset(df, isfinite.(:normalized_statistic))

    # group by site and remove the site mean
    df_clean = @transform(groupby(df_clean, [:site, :variable]), :normalized_statistic = :normalized_statistic .- mean(:normalized_statistic))

    # pivot wide by site and member which are the samples
    wide = unstack(df_clean, [:site, :member], :variable, :normalized_statistic)

    # compute the covariance across the de-meaned sites
    X = Matrix(wide[:, vars])
    Σ = cov(X)

    return Σ
end

function gradient_approximation(df, variables, constrained_params, param_ordering, ::Linear)
    param_df = DataFrame(constrained_params', Symbol.(param_ordering))
    param_df.member = ["member_$(lpad(i, 3, "0"))" for i in 1:size(constrained_params, 2)]
    df_joined = leftjoin(df, param_df, on = :member)

    # get regression formula
    parameter_name_joined = join(param_ordering, "+")
    formula_str = "normalized_statistic ~ $parameter_name_joined + fe(site)"
    regression_formula = eval(Meta.parse("@formula($formula_str)"))

    βs = []
    for (i, var) in enumerate(variables)
        g = filter(:variable => ==(var), df_joined)

        try
            model = reg(g, regression_formula)
            push!(βs, coef(model))
        catch e
            if e isa InterruptException
                throw(e)
            else
                println("$var failed: $e")
                push!(βs, repeat([NaN], len(param_ordering)))
            end
        end
        if i % 50 == 0
            @info "Processed $i/$(length(variables)) variables"
        end
    end

    return hcat(βs...)
end

"""
    normalize_statistics(df)

Filters out NaN statistics and z-score normalizes `df.statistic` per
variable, writing the result to `df.normalized_statistic`. Pure dataframe
transform (no I/O).
"""
function normalize_statistics(df::DataFrame)
    @info "Filtering out NaNs"
    df = df[.!isnan.(df.statistic), :]

    stats_unique = collect(Set(df.variable))
    normalization_dict = Dict()

    for stat in stats_unique
        stats = df[df.variable .== stat, :statistic]
        mean_stat = mean(stats)
        std_stat = std(stats)
        if std_stat == 0 # handle case where the standard deviation is zero
            std_stat = 1
            mean_stat = 0
        end
        normalization_dict[stat] = (mean_stat, std_stat)
    end

    @info "Normalizing statistics"
    means = [normalization_dict[var][1] for var in df.variable]
    stds = [normalization_dict[var][2] for var in df.variable]
    df.normalized_statistic = (df.statistic .- means) ./ stds
    @info "Normalized statistics"

    return df
end

"""
    normalize_constrained_parameters(param_dicts, prior)

Given a vector of already-loaded raw parameter dictionaries (one per
ensemble member - e.g. from `load_member_parameters`, or your own CSV) and an
already-loaded EKP `prior`, transform to unconstrained space and z-score
normalize across members.

This is the piece to call directly if you already have parameters in memory
and don't want to touch the `output_5_cfsites/member_NNN/parameter.toml`
directory layout at all.
"""
function normalize_constrained_parameters(param_dicts, prior)
    constrained_params = []
    parameter_names = []
    for (i, param_dict) in enumerate(param_dicts)
        param_flattened = vcat([param_dict[key] for key in prior.name]...)
        constrained_member_params = EKP.transform_constrained_to_unconstrained(prior, param_flattened) # by constraining we normalize the parameters
        push!(constrained_params, constrained_member_params)

        if i == 1
            parameter_names = vcat([param_dict[name] isa Vector ? [name * "_" * string(j) for j in 1:length(param_dict[name])] : [name] for name in prior.name]...)
        end
    end
    constrained_params_mat = hcat(constrained_params...)

    # normalize the parameters now that they are roughly gaussian
    norm_constrained_params = (constrained_params_mat .- mean(constrained_params_mat, dims=2)) ./ std(constrained_params_mat, dims=2)

    return norm_constrained_params, parameter_names
end

"""
    normalize_parameter_dict(param_values, param_stats)

Normalize a single member's raw parameter `Dict` (as loaded by
`load_raw_parameter_toml` in parameter_io.jl) using precomputed per-parameter
mean/std statistics (as returned by `compute_parameter_statistics`).
"""
function normalize_parameter_dict(param_values::Dict, param_stats::DataFrame)
    normalized_params = Dict()

    for (param_name, value) in param_values
        if value isa Vector
            param_means = param_stats[param_stats.param_name .== param_name, :mean_values][1]
            param_stds = param_stats[param_stats.param_name .== param_name, :std_values][1]
            normalized_params[param_name] = (value .- param_means) ./ param_stds
        else
            param_mean = param_stats[param_stats.param_name .== param_name, :mean_values][1][1]
            param_std = param_stats[param_stats.param_name .== param_name, :std_values][1][1]
            normalized_params[param_name] = (value - param_mean) / param_std
        end
    end

    return normalized_params
end

"""
    compute_parameter_statistics(prior; n_samples=10_000)

Sample from an already-loaded EKP `prior` and compute mean/std statistics per
parameter batch, in constrained space. See experiment.jl for a
`prior_path::String`-taking wrapper.
"""
function compute_parameter_statistics(prior; n_samples::Int = 10_000)
    samples = EKP.transform_unconstrained_to_constrained(
        prior,
        EKP.sample(prior, n_samples)
    )

    sample_means = mean(samples, dims=2)
    sample_stds = std(samples, dims=2)

    param_names = EKP.get_name(prior)
    batch_indices = EKP.batch(prior)

    grouped_means = [sample_means[indices] for indices in batch_indices]
    grouped_stds = [sample_stds[indices] for indices in batch_indices]

    return DataFrame(
        param_name = param_names,
        mean_values = grouped_means,
        std_values = grouped_stds
    )
end

# Builds the flat list of variable names (profile vars expanded across
# z-levels, plus integrated vars) for a given resolution.
function create_var_list(var_names_prof, var_names_int, z_levels)
    all_vars = String[]
    for var in var_names_prof
        start, step, stop = z_levels
        for zlev in collect(start:step:stop)
            push!(all_vars, join([var, zlev], "_"))
        end
    end
    all_vars = vcat(all_vars, var_names_int)
    return all_vars
end

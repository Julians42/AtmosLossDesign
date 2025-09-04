# define types for different types of information gain quantification
abstract type GradientApproximationMethod end
struct Linear <: GradientApproximationMethod end
struct GP <: GradientApproximationMethod end

function information_gain(variables, df, grad_method::GradientApproximationMethod)
    df_sub = df[in.(df.variable, Ref(Set(variables))), :]
    constrained_params, param_ordering = constrained_and_normalized_parameters(rootdir = "output_5_cfsites")
    Σ_0 = cov(constrained_params') # d x d matrix 
    Σ_y = observational_covariance(df_sub, variables) # m x m matrix
    ∇G = gradient_approximation(df_sub, variables, constrained_params, param_ordering, grad_method)'

    # compute information gain relative to the prior
    ig = 1 / 2 * logdet(transpose(∇G) * pinv(Σ_y) * ∇G * Σ_0 + I)
    return ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering
end

function information_gain_plus(variables_plus, variables, df, grad_method::GradientApproximationMethod)
    df_sub = df[in.(df.variable, Ref(Set(variables))), :]
    df_sub_plus = df[in.(df.variable, Ref(Set(variables_plus))), :]

    constrained_params, param_ordering = constrained_and_normalized_parameters(rootdir = "output_5_cfsites")
    Σ_0 = cov(constrained_params)

    Σ_y = observational_covariance(df_sub, variables)
    Σ_y_plus = observational_covariance(df_sub, variables_plus)

    ∇G = gradient_approximation(df_sub, variables, constrained_params, param_ordering, grad_method)
    ∇G_plus = gradient_approximation(df_sub_plus, variables, constrained_params, param_ordering, grad_method)
    

    # compute information gain of the added observation using the information over the posterior
    ig = 1 / 2 * log(det(transpose(∇G_plus) * pinv(Σ_y_plus) * ∇G_plus + inv(Σ_0)) / det(transpose(∇G) * inv(Σ_y) * ∇G + inv(Σ_0)))
    return ig, ∇G, ∇G_plus, Σ_y, Σ_y_plus, Σ_0
end

"""
    constrained_and_normalized_parameters(n_members, rootdir, prior_path)

Loads the ensemble of parameters and transforms them to standard gaussian.
"""
function constrained_and_normalized_parameters(; n_members::Int = 100, 
    rootdir::String = "output_5_cfsites",
    prior_path::String = "priors/prior_diagnostic_pi_entr_smooth_entr_detr_coarse_amip_new.toml"
)
    # transform the parameters to roughly gaussian
    prior = CAL.get_prior(prior_path)
    constrained_params = []
    parameter_names = []
    for member in 1:n_members
        member_dir = joinpath(rootdir, "member_" * lpad(member, 3, "0"))
        param_dict = parameter_values_from_toml(joinpath(member_dir, "parameter.toml"))
        param_flattened = vcat([param_dict[key] for key in prior.name]...)
        constrained_member_params = EKP.transform_constrained_to_unconstrained(prior, param_flattened) # by constraining we normalize the parameters
        push!(constrained_params, constrained_member_params)

        if member == 1
            parameter_names = vcat([param_dict[name] isa Vector ? [name * "_" * string(i) for i in 1:length(param_dict[name])] : [name] for name in prior.name]...)
        end
    end
    constrained_params_mat = hcat(constrained_params...)

    # normalize the parameters now that they are roughly gaussian 
    norm_constrained_params = (constrained_params_mat .- mean(constrained_params_mat, dims=2)) ./ std(constrained_params_mat, dims=2)

    return norm_constrained_params, parameter_names
end

"""
    parameter_values_from_toml(fpath)

Computes the parameter values for an ensemble member. Helper function for `constrained_and_normalized_parameters`

"""
function parameter_values_from_toml(fpath)
    param_dict = TOML.parsefile(fpath)
    values_dict = Dict()
    for (key, value) in param_dict
        values_dict[key] = value["value"]
    end
    return values_dict
end

"""
    Σ_y(df)

Computes the observational covariance over all the samples and sites
"""
function observational_covariance(df, vars)
    #df_clean = filter(r -> !ismissing(r.statistic) && isfinite(r.statistic) && r.variable != "NaN", df)


    # 1) Keep finite stats and valid variable names
    df_clean = filter(r -> !ismissing(r.statistic) && isfinite(r.statistic) && r.variable != "NaN", df)

    # 2) Pivot wide: each row = (site, member), each column = variable
    wide = unstack(df_clean, [:site, :member], :variable, :statistic)

    # 3) Drop rows with any missing across variable columns
    dropmissing!(wide, vars)

    # 4) Data matrix: rows = observations (all sites × members), cols = variables in specified order
    X = Matrix(wide[:, vars]) 

    # 5) Pooled covariance across all observations (columns are variables)
    Σ = cov(X) 

    return Σ
end


function gradient_approximation(df, variables, constrained_params, param_ordering, ::Linear)

    param_df = DataFrame(constrained_params', Symbol.(param_ordering))
    param_df.member = ["member_$(lpad(i, 3, "0"))" for i in 1:size(constrained_params, 2)]
    df_joined = leftjoin(df, param_df, on = :member)

    # get regression formula 
    parameter_name_joined = join(param_ordering, "+")
    formula_str = "statistic ~ $parameter_name_joined + fe(site)"
    regression_formula = eval(Meta.parse("@formula($formula_str)"))

    βs = []
    # for each variable we run linear regression to estimate the gradient
    # for (i, var) in enumerate(variables) 
    #     try 
    #         # filter for the specific variable and compute the coefficients
    #         df_var = filter(:variable => ==(var), df_joined)
    #         model = reg(df_var, regression_formula)
    #         push!(βs, coef(model))
    #     catch e # error handling for cases where the all variables are zero for example
    #         if e isa InterruptException
    #             throw(e) # make sure we can escape with interrupt
    #         else 
    #             println("$var failed: $e")
    #         end
    #     end
    #     if i % 50 ==0 # progress reporting...
    #         @info "Processed $i/$(length(variables)) variables"
    #     end
    # end

    for (i, g) in enumerate(groupby(df_joined, :variable, sort = false))
        var = g.variable[1]

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

abstract type AbstractConfig end
struct Config_cfsites <: AbstractConfig end
struct Config_cfsites_deep <: AbstractConfig end
struct Config_cfsites_shallow <: AbstractConfig end
function get_all_variables(config::Dict, ::Config_cfsites_deep)
    all_vars = String[]
    for var in config["var_names_prof"]
        start, step, stop = config["z_levels"]["deep"]
        for zlev in collect(start:step:stop)
            push!(all_vars, join([var, zlev], "_"))
        end
    end
    all_vars = vcat(all_vars, config["var_names_int"])
    return all_vars
end
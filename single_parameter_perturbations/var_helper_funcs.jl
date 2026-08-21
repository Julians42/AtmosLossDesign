




# extract vector for a given site
function gs(site, params_ordered, grad_df = grad_df)
    v = Vector{Float32}(undef, length(params_ordered))
    for (i, param) in enumerate(params_ordered)
        try
            v[i] = filter(row -> (row.param == param) && (row.site ==site), grad_df).dcre_dparam[1]
        catch e 
            println(e)
            v[i] = 0
        end
    end
    return v
end

# get the parameter covariance matrix 
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
    g(params_ordered, csv_path)

Computes the gradient of CRE wrt each parameter, returns a df
"""
function g(params_ordered, csv_path = "dataframes/spp1_cfsites_intersect2.csv", lat_threshold = 30.)
    # load csv
    df = CSV.read(csv_path, DataFrame)
    sites = unique(df.site)

    dcre_df = DataFrame(param = String[], site = String[], dcre_dparam = Float64[])
    for param in params_ordered
        for site in sites
            # compute the gradient
            dcre = _dcre_dparam(df, site, param)
            push!(dcre_df, (param, site, dcre))
        end
    end
    # filter nan entries
    filter!(:dcre_dparam => isfinite, dcre_df)

    # filter to only tropical low clouds 
    t = transform(dcre_df, :site => ByRow(x -> split(x, "_")) => [:lat, :lon, :site_date])
    transform!(t, [:lat, :lon, :site_date] => ByRow((lat, lon, site_date) -> (parse(Float64, lat), parse(Float64, lon), convert(String, site_date))) => [:lat, :lon, :site_date])
    filter!(row -> (abs(row.lat) <= lat_threshold), t)

    t
end

"""
    _dcre_dparam(df, site, param)

Computes the difference across the IQR. 
"""
function _dcre_dparam(df, site, param)
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

function get_observations(config)
    df = DataFrame(param = String[], site = String[], observation = String[], low_0K = Float64[], high_0K = Float64[], low_4K = Float64[], high_4K = Float64[])
    # get sites
    sites = Set(basename.(glob(config["output_dir_0K"] * "/*/*/*")))
    pop!(sites, "parameters.toml")
    sites = collect(sites)

    # data vars
    data_vars = vcat(config["var_names_int"], config["var_names_prof"])

    # get levels to look over for profile variables 
    start, stp, stop = config["z_levels"]["deep"] # deep and shallow are the same right now
    z_levels = collect(start:stp:stop)

    # get parameters to look over 
    param_names = basename.(glob(config["output_dir_0K"] * "/*"))

    # compute for integrated quantities
    for site in sites
        try
            for obs in data_vars
                for param in param_names
                    if strip(obs) in config["var_names_int"]
                        # integrated variable
                        low_0K = get_observation(param, obs, site, "0K", 0.25, config)
                        high_0K = get_observation(param, obs, site, "0K", 0.75, config)
                        low_4K = get_observation(param, obs, site, "4K", 0.25, config)
                        high_4K = get_observation(param, obs, site, "4K", 0.75, config)

                        push!(df, (param, site, obs, low_0K, high_0K, low_4K, high_4K))
                    else
                        var_zs = obs .* "_" .*(string.(z_levels))
                        # profile variable

                        low_0K_ar = get_observation(param, obs, z_levels, site, "0K", 0.25, config)
                        high_0K_ar = get_observation(param, obs, z_levels, site, "0K", 0.75, config)
                        low_4K_ar = get_observation(param, obs, z_levels, site, "4K", 0.25, config)
                        high_4K_ar = get_observation(param, obs, z_levels, site, "4K", 0.75, config)

                        for (i, zlev) in enumerate(z_levels)
                            push!(df, 
                                (param, 
                                site, 
                                var_zs[i], 
                                low_0K_ar[i], 
                                high_0K_ar[i], 
                                low_4K_ar[i], 
                                high_4K_ar[i])
                            )
                        end
                    end
                end
            end
            println("Processed site $site")
        catch e
            if e isa InterruptException
                throw(e)
            else
                println("Site $site failed: $e")
            end
        end
    end
    return df
end

function get_observation(param, obs, site, scenario, perturb, config)
    # get directory
    if scenario == "0K" 
        output_dir = config["output_dir_0K"]
    else
        output_dir = config["output_dir_4K"]
    end
    simdir_path = joinpath(output_dir, param, string(perturb), site, "output_active")
    simdir = SimDir(simdir_path)

    data = ClimaAnalysis.get(simdir; short_name = obs, reduction = "inst")

    if data.dims["time"][end] < config["reduction_end_time"]
        return NaN
    else
        stat = window(data, "time"; left = config["reduction_start_time"], right = config["reduction_end_time"])
        stat = average_time(slice(stat, x = 0, y = 0))
        return stat.data[1]
    end
end

function get_observation(param, obs, zlev, site, scenario, perturb, config)
    # get directory
    if scenario == "0K" 
        output_dir = config["output_dir_0K"]
    else
        output_dir = config["output_dir_4K"]
    end
    simdir_path = joinpath(output_dir, param, string(perturb), site, "output_active")
    simdir = SimDir(simdir_path)

    data = ClimaAnalysis.get(simdir; short_name = obs, reduction = "inst")

    if data.dims["time"][end] < config["reduction_end_time"]
        return repeat([NaN], length(zlev))
    else
        obs_vec = []
        stat = window(data, "time"; left = config["reduction_start_time"], right = config["reduction_end_time"])
        stat = average_time(slice(stat, x = 0, y = 0))
        for z in zlev
            stat_z = slice(stat, z = z).data[1]
            push!(obs_vec, stat_z)
        end
        return obs_vec
    end
end



# helper function to create the list of variables
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

function cholesky_solve(Σ_y, b)
    Σ_y_reg = Σ_y + 1e-10 * I
    if !issymmetric(Σ_y)
        Σ_y_reg = 0.5 * (Σ_y_reg + Σ_y_reg')
    end
    L = cholesky(Σ_y_reg).L
    return L' \ (L \ b)
end


function observational_posterior_covariance(Σ₀, vars, df, params_ordered)
    # clean data 
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

    df_wide = unstack(df_clean, [:param, :site, :observation], :scenario, :value)
    df_wide.grad_0K = (df_wide.high_0K .- df_wide.low_0K)
    df_wide.grad_4K = (df_wide.high_4K .- df_wide.low_4K)

    Q_grad_0K = df_wide[:, [:param, :site, :observation, :grad_0K]]
    Q_grad_4K = df_wide[:, [:param, :site, :observation, :grad_4K]]
    dropmissing!(Q_grad_0K, :grad_0K)
    dropmissing!(Q_grad_4K, :grad_4K)
    Q_grad_0K = @subset(Q_grad_0K, isfinite.(:grad_0K))
    Q_grad_4K = @subset(Q_grad_4K, isfinite.(:grad_4K))

    # select observations in vars and average over sites and then pivot wider 
    Q_grad_0K = @subset(Q_grad_0K, :observation .∈ Ref(vars))
    Q_grad_0K = combine(groupby(Q_grad_0K, [:param, :observation]), :grad_0K => mean => :grad_0K)
    Q_grad_0K_wide = unstack(Q_grad_0K, [:param], :observation, :grad_0K)
    order_indices_0K = [findfirst(==(p), Q_grad_0K_wide.param) for p in params_ordered if p in Q_grad_0K_wide.param]
    Q_mat_0K = Matrix(Q_grad_0K_wide[order_indices_0K, vars])

    Q_grad_4K = @subset(Q_grad_4K, :observation .∈ Ref(vars))
    Q_grad_4K = combine(groupby(Q_grad_4K, [:param, :observation]), :grad_4K => mean => :grad_4K)
    Q_grad_4K_wide = unstack(Q_grad_4K, [:param], :observation, :grad_4K)
    order_indices_4K = [findfirst(==(p), Q_grad_4K_wide.param) for p in params_ordered if p in Q_grad_4K_wide.param]
    Q_mat_4K = Matrix(Q_grad_4K_wide[order_indices_4K, vars])


    Σ_θ_post_0K = inv(Q_mat_0K * cholesky_solve(Σ_y, Q_mat_0K') + inv(Σ₀))
    Σ_θ_post_4K = inv(Q_mat_4K * cholesky_solve(Σ_y, Q_mat_4K') + inv(Σ₀))

    return Σ_θ_post_0K, Σ_θ_post_4K
end


# to bootstrap we sample randomly from gs_vectors (considering only uncertainty from gradient estimates of CRE not gradient estimates of covariance estimates)
function get_boot_cre_grad(gs_vectors, n_samples)
    boot_sample_means = []
    for i in 1:n_samples
        sample_indices = rand(1:length(gs_vectors), length(gs_vectors))
        sampled_gs = gs_vectors[sample_indices]
        mat = hcat(sampled_gs...) 
        mat[mat .== 0] .= NaN
        mean_gs = vec(nanmean(mat, dims = 2))
        push!(boot_sample_means, mean_gs)
    end
    return boot_sample_means
end

function gradient_approximation(df, variables, constrained_params, param_ordering)

    param_df = DataFrame(constrained_params', Symbol.(param_ordering))
    param_df.member = ["member_$(lpad(i, 3, "0"))" for i in 1:size(constrained_params, 2)]
    df_joined = leftjoin(df, param_df, on = :member)

    # get regression formula 
    parameter_name_joined = join(param_ordering, "+")
    formula_str = "normalized_statistic ~ $parameter_name_joined + fe(site)"
    regression_formula = eval(Meta.parse("@formula($formula_str)"))

    βs = []
    #for (i, g) in enumerate(groupby(df_joined, :variable, sort = false))
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


function get_all_variables(config::Dict)
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

function cholesky_solve(Σ_y, b)
    Σ_y_reg = Σ_y + 1e-10 * I
    if !issymmetric(Σ_y)
        Σ_y_reg = 0.5 * (Σ_y_reg + Σ_y_reg')
    end
    L = cholesky(Σ_y_reg).L
    return L' \ (L \ b)
end


function get_variance_reduction(prof_vars, int_vars, z_levels, Σ₀, df, params_ordered, gs_vectors)
    boot_gs_means = get_boot_cre_grad(gs_vectors, 1000)

    variance_reduction_dfs = []
    # compute the variance matrices for variables 
    for var in prof_vars
        vars = create_var_list([var], [], z_levels)#config["z_levels"]["shallow"])
        Σ_θ_post_0K, Σ_θ_post_4K = observational_posterior_covariance(Σ₀, vars, df, params_ordered)
        Σ_θ_post = Σ_θ_post_0K #(Σ_θ_post_0K + Σ_θ_post_4K) / 2.0
        # boot_gs_means = get_boot_cre_grad(gs_vectors, 1000)
        Σ_cre_pre_bs = [mean_gs' * Σ₀ * mean_gs for mean_gs in boot_gs_means]
        Σ_cre_post_bs = [mean_gs' * Σ_θ_post * mean_gs for mean_gs in boot_gs_means]
        df_var = DataFrame(variable = repeat([var], length(boot_gs_means)),
                            prior_var = Σ_cre_pre_bs,
                            post_var = Σ_cre_post_bs,
                            reduction =  (Σ_cre_pre_bs .- Σ_cre_post_bs) ./ Σ_cre_pre_bs) 
        push!(variance_reduction_dfs, df_var)
        println(var, "  ", mean((Σ_cre_post_bs .- Σ_cre_pre_bs) ./ Σ_cre_pre_bs))
    end

    for var in int_vars
        # vars = create_var_list([var], [], config["z_levels"]["shallow"])
        Σ_θ_post_0K, Σ_θ_post_4K = observational_posterior_covariance(Σ₀, [var], df, params_ordered)
        Σ_θ_post = Σ_θ_post_0K#(Σ_θ_post_0K + Σ_θ_post_4K) / 2.0

        Σ_cre_post_mean = mean_gs' * Σ_θ_post * mean_gs

        #boot_gs_means = get_boot_cre_grad(gs_vectors, 1000)

        Σ_cre_pre_bs = [mean_gs' * Σ₀ * mean_gs for mean_gs in boot_gs_means]
        Σ_cre_post_bs = [mean_gs' * Σ_θ_post * mean_gs for mean_gs in boot_gs_means]

        df_var = DataFrame(variable = repeat([var], length(boot_gs_means)),
                        prior_var = Σ_cre_pre_bs,
                        post_var = Σ_cre_post_bs,
                        reduction =  (Σ_cre_pre_bs .- Σ_cre_post_bs) ./ Σ_cre_pre_bs) 
        push!(variance_reduction_dfs, df_var)
        println(var, "  ", mean((Σ_cre_post_bs .- Σ_cre_pre_bs) ./ Σ_cre_pre_bs))
    end

    df_variance_reduction = vcat(variance_reduction_dfs...)

    summaries = combine(groupby(df_variance_reduction, :variable), :reduction => Statistics.median => :mean_reduction)
    sort!(summaries, :mean_reduction, rev = true)
    df_variance_reduction.variable = CategoricalArray(df_variance_reduction.variable, levels = summaries.variable, ordered = true)
    return df_variance_reduction
end





# vars = create_var_list(["ta"], [], config["z_levels"]["shallow"])


# df_clean = @subset(df, isfinite.(:low_0K) .& isfinite.(:high_0K) .& isfinite.(:low_4K) .& isfinite.(:high_4K))
# df_clean = stack(df, [:low_0K, :high_0K, :low_4K, :high_4K], variable_name = :scenario, value_name = :value)
# df_clean = @subset(df_clean, isfinite.(:value))
# wide = unstack(df_clean, [:param, :site, :scenario], :observation, :value)
# dropmissing!(wide)
# X = Matrix(wide[:, vars])
# Σ_y = cov(X)

# # obs_list = []

# # compute grad_Q: for each parameter and variable in observation set compute 1 / (high_0K - low_0K) and then 
# # put in the \nabla_Q matrix

# df_wide = unstack(df_clean, [:param, :site, :observation], :scenario, :value)
# df_wide.grad_0K = (df_wide.high_0K .- df_wide.low_0K)
# df_wide.grad_4K = (df_wide.high_4K .- df_wide.low_4K)

# Q_grad_0K = df_wide[:, [:param, :site, :observation, :grad_0K]]
# dropmissing!(Q_grad_0K, :grad_0K)
# Q_grad_0K = @subset(Q_grad_0K, isfinite.(:grad_0K))

# # select observations in vars and average over sites and then pivot wider 
# Q_grad_0K = @subset(Q_grad_0K, :observation .∈ Ref(vars))
# Q_grad_0K = combine(groupby(Q_grad_0K, [:param, :observation]), :grad_0K => mean => :grad_0K)
# Q_grad_0K_wide = unstack(Q_grad_0K, [:param], :observation, :grad_0K)
# order_indices_0K = [findfirst(==(p), Q_grad_0K_wide.param) for p in params_ordered if p in Q_grad_0K_wide.param]
# Q_mat_0K = Matrix(Q_grad_0K_wide[order_indices_0K, vars])
"""
    compute_parameter_statistics(prior_path::String; n_samples::Int=10_000)

Compute mean and standard deviation statistics for parameters defined in a prior file.

# Arguments
- `prior_path::String`: Path to the prior TOML file
- `n_samples::Int=10_000`: Number of samples to use for computing statistics

# Returns
- `DataFrame`: Contains parameter names and their corresponding mean and standard deviation values
    grouped by parameter batches

# Steps:
1. Load prior from TOML file
2. Generate samples from prior distributions
3. Transform samples from unconstrained to constrained space
4. Compute mean and standard deviation of samples
5. Group statistics by parameter batches
6. Return results as DataFrame
"""
function compute_parameter_statistics(prior_path::String; n_samples::Int=10_000)
    # Load prior from file
    prior = CAL.get_prior(prior_path)
    
    # Generate and transform samples
    samples = EKP.transform_unconstrained_to_constrained(
        prior, 
        EKP.sample(prior, n_samples)
    )
    
    # Compute statistics
    sample_means = mean(samples, dims=2)
    sample_stds = std(samples, dims=2)
    
    # Get parameter names and batch indices
    param_names = EKP.get_name(prior)
    batch_indices = EKP.batch(prior)
    
    # Group statistics by parameter batches
    grouped_means = [sample_means[indices] for indices in batch_indices]
    grouped_stds = [sample_stds[indices] for indices in batch_indices]
    
    # Create and return DataFrame
    return DataFrame(
        param_name = param_names,
        mean_values = grouped_means,
        std_values = grouped_stds
    )
end

function get_param_values(fpath)
    param_dict = TOML.parsefile(fpath)
    values_dict = Dict()
    for (key, value) in param_dict
        values_dict[key] = value["value"]
    end
    return values_dict
end

"""
    normalize_parameters(fpath::String, param_stats::DataFrame)

Normalize parameter values from a TOML file using pre-computed parameter statistics.

# Arguments
- `fpath::String`: Path to parameter TOML file
- `param_stats::DataFrame`: Pre-computed parameter statistics from compute_parameter_statistics()

# Returns
- `Dict`: Dictionary of normalized parameter values
"""
function normalize_parameters(fpath::String, param_stats::DataFrame)
    # Get raw parameter values
    param_values = get_param_values(fpath)
    
    normalized_params = Dict()
    
    for (param_name, value) in param_values
        if value isa Vector
            # Handle vector parameters
            param_means = param_stats[param_stats.param_name .== param_name, :mean_values][1]
            param_stds = param_stats[param_stats.param_name .== param_name, :std_values][1]
            normalized_params[param_name] = (value .- param_means) ./ param_stds
        else
            # Handle scalar parameters
            param_mean = param_stats[param_stats.param_name .== param_name, :mean_values][1][1]
            param_std = param_stats[param_stats.param_name .== param_name, :std_values][1][1]
            normalized_params[param_name] = (value - param_mean) / param_std
        end
    end
    
    return normalized_params
end

function flatten_dict_column!(df::DataFrame, dict_col::Symbol)
    # Initialize arrays to store flattened values and keys
    val_arrays = []
    key_arrays = []
    
    # Process each row's dictionary
    for row in eachrow(df)
        val_array = []
        key_array = []
        
        # Flatten the dictionary in this row
        for (key, value) in row[dict_col]
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
        
        push!(val_arrays, val_array)
        push!(key_arrays, key_array)
    end
    
    # Add new columns to dataframe
    df[!, Symbol(dict_col, "_values")] = val_arrays
    df[!, Symbol(dict_col, "_keys")] = key_arrays
    
    return df
end

function create_parameter_dataframe(param_stats_df; num_members=100)
    # Create empty arrays to store values and keys for each member
    all_vals = []
    all_keys = []
    members = []

    # Loop through all members
    for i in 1:num_members
        member = "member_" * lpad(i, 3, "0")
        push!(members, member)
        
        # Get normalized parameters for this member
        norm_params = normalize_parameters("output_2/$member/parameter.toml", param_stats_df)
        
        val_array = []
        key_array = []
        
        # Flatten the dictionary
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

    # Create dataframe
    parameter_df = DataFrame(member = members)
    parameter_df[!, :values] = all_vals
    parameter_df[!, :keys] = all_keys
    
    return parameter_df
end


function postprocess_dataframe(df::DataFrame)
    df = filter(row -> !isnan(row.statistic), df)

    stats_unique = collect(Set(df.variable))

    normalization_dict = Dict()

    for stat in stats_unique
        # get statistics of each variable
        stats = filter(row -> row.variable == stat, df).statistic
        # get the mean and std of the statistics
        mean_stat = mean(stats)
        std_stat = std(stats)
        # store the mean and std in the normalization_dict
        normalization_dict[stat] = (mean_stat, std_stat)
    end
    
    # Create a new column for normalized statistics
    df.normalized_statistic = map(row -> begin
        mean_val, std_val = normalization_dict[row.variable]
        (row.statistic - mean_val) / std_val
    end, eachrow(df))

    # merge in the normalized parameter dataframe
    parameter_df = create_parameter_dataframe(param_stats_df, num_members = 100)
    combined_df = innerjoin(df, parameter_df; on=:member)

    return combined_df
end


function process_members_sites(members, sites, param_stats_df, config)
    rows = []

    data_vars = vcat(config["var_names_int"], config["var_names_prof"])

    for member in members
        # Get normalized parameters for this member
        norm_params = normalize_parameters("output_2/$member/parameter.toml", param_stats_df)
        
        for site in sites
            sim_dir = SimDir(joinpath("output_2", member, site, "output_active"))
            try
                for data_var in data_vars
                    data = get(sim_dir; short_name = data_var, reduction = "inst")

                    if data.dims["time"][end] < config["reduction_end_time"]
                        throw()
                    else
                        profile_data = window(data, "time";
                                            left=config["reduction_start_time"],
                                            right=config["reduction_end_time"])
                        averaged_profile = average_time(slice(profile_data, x=0, y=0))

                        if !haskey(averaged_profile.dims, "z")
                            stat = averaged_profile.data[1]
                            push!(rows, (
                                member = member,
                                site = site,
                                variable = data_var,
                                statistic = stat,
                            ))
                        else
                            for zlev in config["z_levels"]
                                stat = slice(averaged_profile, z=zlev).data[1]
                                push!(rows, (
                                    member = member,
                                    site = site,
                                    variable = join([data_var, zlev], "_"),
                                    statistic = stat,
                                ))
                            end
                        end
                    end
                end
            catch
                @info "Simulation failed for $member and $site. Appending NaNs..."
                push!(rows, (
                    member = member,
                    site = site,
                    variable = NaN,
                    statistic = NaN,
                ))
            end
        end
    end

    # Convert rows array to DataFrame and return
    return DataFrame(rows)
end


function get_regression_coefficients(df::DataFrame, variable::String)
    t = filter(row -> row.variable == variable, df)[:, [:member, :site, :normalized_statistic, :values]]
    parameter_df = DataFrame(transpose(hcat(t.values...)), :auto)
    combined_df = hcat(t[:, [:member, :site, :normalized_statistic]], parameter_df)
    test_formula = @formula(normalized_statistic ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + fe(site))
    model = reg(combined_df, test_formula)
    return model.coef
end


"""
    compute_regression_coefficients(df_cleaned::DataFrame, config::Dict)

Compute regression coefficients for all variables in the configuration.
Returns a tuple containing the coefficient array and variable names.
"""
function compute_regression_coefficients(df_cleaned::DataFrame, config::Dict)
    coef_ar = []
    var_names = []
    for var in get_all_variables(config)
        try
            # get regression coefficients
            coef = get_regression_coefficients(df_cleaned, var)
            push!(coef_ar, coef)
            push!(var_names, var)
        catch
            println(var, " failed")
        end
    end
    coef_ar = hcat(coef_ar...)
    return coef_ar, var_names
end

function get_all_variables(config::Dict)
    all_vars =[]
    for var in config["var_names_prof"]
        for zlev in config["z_levels"]
            push!(all_vars, join([var, zlev], "_"))
        end
    end
    all_vars = vcat(all_vars, config["var_names_int"])
    return all_vars
end


"""
    plot_regression_analysis(B_matrix::Matrix, B_names::Vector, param_names::Vector, config::Dict)

Create and save four plots analyzing the regression coefficients matrix B:
1. Scree plot of parameter informed dimension (BBᵀ eigenvalues)
2. Scree plot of variable informative dimension (BᵀB eigenvalues)
3. Bar plot of parameter informed dimension (diagonal of BBᵀ)
4. Bar plot of variable informative dimension (diagonal of BᵀB)

Each plot is saved separately with the experiment name in the filename.
"""
function plot_regression_analysis(B_matrix::Matrix, B_names::Vector, param_names::Vector, config::Dict)
    exp_name = config["exp_name"]
    
    # Plot 1: Scree plot of parameter informed dimension
    fig1 = Figure(size = (500, 400))
    ax1 = Axis(fig1[1,1], 
        xlabel = "Dimension",
        ylabel = "Eigenvalue of BBᵀ",
        title = "Scree plot of parameter informed dimension ($(exp_name))")
    scatter!(ax1, 1:length(svd(B_matrix * transpose(B_matrix)).S), 
             svd(B_matrix * transpose(B_matrix)).S)
    save("plots/$(exp_name)_scree_param_informed.png", fig1)

    # Plot 2: Scree plot of variable informative dimension
    fig2 = Figure(size = (500, 400))
    ax2 = Axis(fig2[1,1], 
        xlabel = "Dimension",
        yscale = log10,
        ylabel = "Eigenvalue of BᵀB",
        title = "Scree plot of variable informative dimension ($(exp_name))")
    scatter!(ax2, 1:length(svd(transpose(B_matrix) * B_matrix).S), 
             svd(transpose(B_matrix) * B_matrix).S)
    save("plots/$(exp_name)_scree_var_informed.png", fig2)

    # Calculate diagonal values
    parameter_informed_diag = diag(B_matrix * transpose(B_matrix))
    variable_informed_diag = diag(transpose(B_matrix) * B_matrix)

    # Plot 3: Parameter informed bar plot
    fig3 = Figure(size = (800, 400))
    ax3 = Axis(fig3[1,1],
        xlabel = "Parameter",
        ylabel = "Diagonal value of BBᵀ",
        title = "Parameter informed dimension ($(exp_name))")

    sorted_param_inds = sortperm(parameter_informed_diag, rev=true)
    sorted_param_vals = parameter_informed_diag[sorted_param_inds]
    sorted_param_names = param_names[sorted_param_inds]

    barplot!(ax3, 1:length(sorted_param_vals), sorted_param_vals)
    ax3.xticks = (1:length(sorted_param_names), sorted_param_names)
    ax3.xticklabelrotation = π/2
    save("plots/$(exp_name)_param_informed.png", fig3)

    # Plot 4: Variable informed bar plot
    fig4 = Figure(size = (800, 400))
    ax4 = Axis(fig4[1,1],
        xlabel = "Variable",
        ylabel = "Diagonal value of BᵀB",
        title = "Variable informative dimension ($(exp_name))")

    sorted_var_inds = sortperm(variable_informed_diag, rev=true)
    sorted_var_vals = variable_informed_diag[sorted_var_inds]
    sorted_var_names = B_names[sorted_var_inds]

    barplot!(ax4, 1:length(sorted_var_vals), sorted_var_vals)
    ax4.xticks = (1:length(sorted_var_names), sorted_var_names)
    ax4.xticklabelrotation = π/2
    save("plots/$(exp_name)_var_informed.png", fig4)

    return fig1, fig2, fig3, fig4
end

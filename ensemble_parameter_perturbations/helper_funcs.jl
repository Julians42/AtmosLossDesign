using LaTeXStrings # for plotting

set_theme!(theme_latexfonts(),
    fontsize = 14,
    fonts = (
        regular = "Latin Modern Roman",
        bold = "Latin Modern Sans Demi Bold",
        italic = "Latin Modern Roman Italic",
        bold_italic = "Latin Modern Roman Bold Italic",
    ),
    Axis = (
        titlefont = "Latin Modern Roman",
        titlesize = 16,
    ),
    Figure = (
        titlefont = :bold,
        titlesize = 18,
    ),
    linewidth = 1.5,
    markersize = 8,
)



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

function create_parameter_dataframe(rootdir, param_stats_df; num_members=100)
    # Create empty arrays to store values and keys for each member
    all_vals = []
    all_keys = []
    members = []

    # Loop through all members
    for i in 1:num_members
        member = "member_" * lpad(i, 3, "0")
        push!(members, member)
        
        # Get normalized parameters for this member
        norm_params = normalize_parameters("$rootdir/$member/parameter.toml", param_stats_df)
        
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


function postprocess_dataframe(df::DataFrame, param_stats_df::DataFrame, num_members::Int, rootdir::String)
    @info "Filtering out NaNs"
    #df = filter(row -> !isnan(row.statistic), df)
    df = df[.!isnan.(df.statistic), :]

    stats_unique = collect(Set(df.variable))

    normalization_dict = Dict()

    for stat in stats_unique
        # get statistics of each variable
        stats = df[df.variable .== stat, :statistic]
        # get the mean and std of the statistics
        mean_stat = mean(stats)
        std_stat = std(stats)
        # store the mean and std in the normalization_dict
        normalization_dict[stat] = (mean_stat, std_stat)
    end
    
    # Create a new column for normalized statistics
    # df.normalized_statistic = map(row -> begin
    #     mean_val, std_val = normalization_dict[row.variable]
    #     (row.statistic - mean_val) / std_val
    # end, eachrow(df))
    @info "Normalizing statistics"
    means = [normalization_dict[var][1] for var in df.variable]
    stds = [normalization_dict[var][2] for var in df.variable]
    @info "Got means and stds"
    df.normalized_statistic = (df.statistic .- means) ./ stds
    @info "Normalized statistics"

    # merge in the normalized parameter dataframe
    parameter_df = create_parameter_dataframe(rootdir,param_stats_df, num_members = num_members)
    combined_df = innerjoin(df, parameter_df; on=:member)

    return combined_df
end


function process_members_sites(members, sites, param_stats_df, config)
    rows = []

    data_vars = vcat(config["var_names_int"], config["var_names_prof"])

    for member in members
        @info "Processing member $member"
        # Get normalized parameters for this member
        norm_params = normalize_parameters("$(config["output_dir"])/$member/parameter.toml", param_stats_df)
        
        for site in sites
            site_data_path = joinpath(config["output_dir"], member, site, "output_active")
            sim_dir = SimDir(site_data_path)
            # get the type of convection from the .yml file
            forcing_type = YAML.load_file(joinpath(site_data_path, ".yml"))
            
            # Determine z_levels based on whether "deep" or "shallow" appears in TOML files
            toml_files = forcing_type["toml"]
            toml_string = join(toml_files, " ")
            if occursin("deep", toml_string)
                start, step, stop = config["z_levels"]["deep"]
                z_levels = collect(start:step:stop)  # z_levels_deep
                convection_type = "deep"
            elseif occursin("shallow", toml_string)
                start, step, stop = config["z_levels"]["shallow"]
                z_levels = collect(start:step:stop)   # z_levels_shallow
                convection_type = "shallow"
            else
                @error "No deep or shallow forcing found in $site_data_path. Revise script..."
                convection_type = "unknown"
                #z_levels = config["z_levels"]  # could fallback to config default
            end
            try
                for data_var in data_vars
                    data = get(sim_dir; short_name = data_var, reduction = "inst")

                    if data.dims["time"][end] < config["reduction_end_time"]
                        throw(ErrorException("Simulation time too short"))
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
                            convection_type = convection_type,
                        ))
                        else
                                                    for zlev in z_levels
                            stat = slice(averaged_profile, z=zlev).data[1]
                            push!(rows, (
                                member = member,
                                site = site,
                                variable = join([data_var, zlev], "_"),
                                statistic = stat,
                                convection_type = convection_type,
                            ))
                            end
                        end
                    end
                end
                println("completed processing $member and $site")
            catch
                @info "Simulation failed for $member and $site. Appending NaNs..."
                push!(rows, (
                    member = member,
                    site = site,
                    variable = NaN,
                    statistic = NaN,
                    convection_type = convection_type,
                ))
            end
        end
    end

    # Convert rows array to DataFrame and return
    return DataFrame(rows)
end


# function get_regression_coefficients(df::DataFrame, variable::String)
#     t = filter(row -> row.variable == variable, df)[:, [:member, :site, :normalized_statistic, :values, :keys]]
#     parameter_df = DataFrame(transpose(hcat(t.values...)), Symbol.(t.keys[1]))
#     combined_df = hcat(t[:, [:member, :site, :normalized_statistic]], parameter_df)
    
#     # Get parameter column names and construct formula dynamically
#     param_cols = names(parameter_df)
#     param_terms = join(param_cols, " + ")
#     formula_str = "normalized_statistic ~ $param_terms + fe(site)"
#     test_formula = eval(Meta.parse("@formula($formula_str)"))
    
#     model = reg(combined_df, test_formula)
#     return model.coef, model.coefnames
# end


# """
#     compute_regression_coefficients(df_cleaned::DataFrame, config::Dict)

# Compute regression coefficients for all variables in the configuration.
# Returns a tuple containing the coefficient array and variable names.
# """
# function compute_regression_coefficients(df_cleaned::DataFrame, informing_variables)
#     coef_ar = []
#     coef_names = []
#     failed_vars = []
#     for (i, var) in enumerate(informing_variables)
#         try
#             # get regression coefficients
#             coef, coef_name = get_regression_coefficients(df_cleaned, var)
#             push!(coef_ar, coef)
#             push!(coef_names, coef_name)
#         catch
#             println(var, " failed")
#             push!(failed_vars, i)
#         end
#     end
#     coef_ar = hcat(coef_ar...)
#     return coef_ar, coef_names[1], failed_vars
# end

function compute_regression_coefficients_optimized(df_cleaned::DataFrame, informing_variables)
    # Pre-group the dataframe by variable (MAJOR speedup)
    @info "Grouping dataframe by variable..."
    grouped_df = groupby(df_cleaned, :variable)
    
    # Pre-compile the formula (avoid repeated parsing)
    # Assume all variables have the same parameter structure
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
            # Fast lookup instead of filtering entire dataframe
            if haskey(grouped_df, (variable = var,))
                var_data = grouped_df[(variable = var,)]
                
                # Efficient matrix construction
                n_rows = nrow(var_data)
                n_params = length(param_cols)
                param_matrix = Matrix{Float64}(undef, n_rows, n_params)
                
                # Vectorized assignment instead of transpose(hcat(...))
                for (row_idx, values) in enumerate(var_data.values)
                    param_matrix[row_idx, :] = values
                end
                
                # Create dataframe efficiently
                parameter_df = DataFrame(param_matrix, param_cols)
                combined_df = hcat(
                    select(var_data, [:member, :site, :normalized_statistic]), 
                    parameter_df
                )
                
                # Reuse pre-compiled formula
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
        
        # Progress indicator for large jobs
        if i % 50 == 0
            @info "Processed $i/$(length(informing_variables)) variables"
        end
    end
    
    coef_ar = hcat(coef_ar...)
    return coef_ar, coef_names[1], failed_vars
end

abstract type AbstractConfig end
struct Config_cfsites <: AbstractConfig end
struct Config_cfsites_deep <: AbstractConfig end
struct Config_cfsites_shallow <: AbstractConfig end


function get_all_variables(config::Dict, ::Config_cfsites)
    all_vars = String[]
    for var in config["var_names_prof"]
        for zlev in config["z_levels"]
            push!(all_vars, join([var, zlev], "_"))
        end
    end
    all_vars = vcat(all_vars, config["var_names_int"])
    return all_vars
end

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

function get_all_variables(config::Dict, ::Config_cfsites_shallow)
    all_vars = String[]
    for var in config["var_names_prof"]
        start, step, stop = config["z_levels"]["shallow"]
        for zlev in collect(start:step:stop)
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
    
    # Plot 1: Scree plot of parameter informed dimension
    fig1 = Figure(size = (500, 400))
    ax1 = Axis(fig1[1,1], 
        xlabel = "Principle Component Dimension",
        xlabelsize = 12,
        ylabel = L"\text{Eigenvalue of } H_\theta H_\theta^T",
        title = "Learnable Dimension")
    scatter!(ax1, 1:length(svd(B_matrix * transpose(B_matrix)).S), 
             svd(B_matrix * transpose(B_matrix)).S)
    save("plots/$(config["exp_name"])_scree_param_informed.png", fig1)

    # # Plot 2: Scree plot of variable informative dimension
    # fig2 = Figure(size = (500, 400))
    # ax2 = Axis(fig2[1,1], 
    #     xlabel = "Dimension",
    #     yscale = log10,
    #     ylabel = "Eigenvalue of BᵀB",
    #     title = "Scree plot of variable informative dimension ($(exp_name))")
    # scatter!(ax2, 1:length(svd(transpose(B_matrix) * B_matrix).S), 
    #          svd(transpose(B_matrix) * B_matrix).S)
    # save("plots/$(exp_name)_scree_var_informed.png", fig2)

    # Calculate diagonal values
    parameter_informed_diag = diag(B_matrix * transpose(B_matrix))
    variable_informed_diag = diag(transpose(B_matrix) * B_matrix)

    # Plot 3: Parameter informed bar plot
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
    save("plots/$(config["exp_name"])_param_informed.png", fig3)

    # Plot 4: Variable informed bar plot
    fig4 = Figure(size = (800, 400))
    ax4 = Axis(fig4[1,1],
        xlabel = "Variable",
        ylabel = "Diagonal value of BᵀB",
        title = "Variable informative dimension ($(config["exp_name"]))")

    sorted_var_inds = sortperm(variable_informed_diag, rev=true)
    sorted_var_vals = variable_informed_diag[sorted_var_inds]
    sorted_var_names = B_names[sorted_var_inds]

    # filter to variables with sufficient information content relative to the most informative variable
    important_var_inds = sorted_var_vals .> 0.05 * sorted_var_vals[1]
    sorted_var_vals = sorted_var_vals[important_var_inds]
    sorted_var_inds = sorted_var_inds[important_var_inds]
    sorted_var_names = sorted_var_names[important_var_inds]

    barplot!(ax4, 1:length(sorted_var_vals), sorted_var_vals)
    ax4.xticks = (1:length(sorted_var_names), sorted_var_names)
    ax4.xticklabelrotation = π/2
    save("plots/$(config["exp_name"])_var_informed.png", fig4)

    plot_variable_informing_analysis(B_matrix, B_names, 
                                     config["var_names_prof"], config["var_names_int"], config["exp_name"])

    return sorted_var_vals, sorted_var_inds, sorted_var_names
end

"""
    plot_variable_informing_analysis(reg_coefs::Matrix, informing_variables::Vector{String}, 
                                   var_names_prof::Vector{String}, var_names_int::Vector{String}, 
                                   exp_name::String)

Create and save a bar plot showing variable informing values sorted by magnitude.
This function preprocesses the regression coefficients to compute integrated variable informing values.

# Arguments
- `reg_coefs::Matrix`: Regression coefficients matrix
- `informing_variables::Vector{String}`: Vector of variable names
- `var_names_prof::Vector{String}`: Profile variable names (to be integrated across levels)
- `var_names_int::Vector{String}`: Integrated variable names (single values)
- `exp_name::String`: Experiment name for the plot title and filename

# Returns
- Nothing, saves plot to "plots/\$(exp_name)_var_informing_integrated.png"
"""
function plot_variable_informing_analysis(reg_coefs::Matrix, informing_variables::Vector{String}, 
                                       var_names_prof::Vector{String}, var_names_int::Vector{String}, 
                                       exp_name::String)
    var_informing_ar = []
    
    # Process profile variables (integrate across levels)
    for profile_var in var_names_prof
        matching_indx = getindex.(split.(informing_variables, "_"), 1) .== profile_var
        # get variable informed array
        var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
        # Main.@infiltrate
        # println(profile_var, " ", tr(var_informed_ar))
        # println(profile_var, " ", sum(matching_indx))
        #push!(var_informing_ar, [profile_var, tr(var_informed_ar)])
        println(profile_var)
        push!(var_informing_ar, [profile_var, - 1/2 *log(abs(det(I - var_informed_ar))) / size(var_informed_ar, 1)])
    end

    # Process integrated variables (single values)
    for single_var in var_names_int
        try
            matching_indx = getindex.(split.(informing_variables, "_"), 1) .== single_var
            # get variable informed array
            var_informed_ar = transpose(reg_coefs[:, matching_indx]) * reg_coefs[:, matching_indx]
            println(single_var, " ", var_informed_ar[1,1])
            println(single_var, " ", sum(matching_indx))
            push!(var_informing_ar, [single_var, -1/2 * log(1- var_informed_ar[1,1])])
        catch
            println(single_var, " failed")
        end

    end

    # Convert the array to a proper DataFrame
    var_informing_matrix = hcat(var_informing_ar...)
    var_informing_df = DataFrame(
        var_name = var_informing_matrix[1, :],
        var_informing = var_informing_matrix[2, :]
    )
    
    # Sort by var_informing magnitude (descending)
    sorted_df = sort(var_informing_df, :var_informing, rev=true)
    
    # Create bar plot of variable informing dimension
    fig_var_informing = Figure(size = (800, 400))
    ax_var_informing = Axis(fig_var_informing[1,1],
        xlabel = "Observation",
        ylabel = "Normalized Observation Informativeness",
        title = "Observation Informativeness for Diagnostic SCM")
    
    barplot!(ax_var_informing, 1:length(sorted_df.var_informing), sorted_df.var_informing ./ norm(sorted_df.var_informing, 1))
    ax_var_informing.xticks = (1:length(sorted_df.var_name), sorted_df.var_name)
    ax_var_informing.xticklabelrotation = π/2
    save("plots/$(exp_name)_var_informing_integrated.png", fig_var_informing)
end


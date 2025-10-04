# helper functions for single parameter perturbations

"""
    get_perturbed_toml(param_name, q;
                            default_toml = default_toml, 
                            q_mat = quantile_matrix, 
                            param_names = param_names,
                            quantiles = quantiles,
)
    Returns a perturbed toml file for the specified parameter name and quantile value.
"""
function get_perturbed_toml(param_name, q;
                            default_toml = default_toml, 
                            q_mat = quantile_matrix, 
                            param_names = param_names,
                            quantiles = quantiles,
)
    perturbed_toml = deepcopy(default_toml)

    # get the matrix indices to extract the perturbed value
    param_index = findfirst(==(param_name), param_names)
    quantile_index = findfirst(==(q), quantiles)

    # assign the value to the correct entry of the toml
    if haskey(perturbed_toml, param_name[1:end-2]) # then its a vector of parameters 
        vec_param_idx = parse(Int, split(param_name, "_")[end])
        perturbed_toml[param_name[1:end-2]]["value"][vec_param_idx] = q_mat[param_index, quantile_index]
    else # single scalar 
        perturbed_toml[param_name]["value"] = q_mat[param_index, quantile_index]
    end
    return perturbed_toml
end

"""
    get_flat_names(prior)
    
Returns a list of parameter names including the flattened names for vector-based parameters.
"""
function get_flat_names(prior)  
    param_names = []
    for (i, name) in enumerate(prior.name)
        if prior.distribution[i].distribution isa Vector
            for j in 1:length(prior.distribution[i].distribution)
                push!(param_names, "$(name)_$j")
            end
        else
            push!(param_names, name)
        end
    end
    return param_names
end

"""
    param_perturbations(prior_path::String; n_samples::Int=10_000, quantiles = 0.1:0.1:0.9)
    
Returns a matrix of parameter values and a list of parameter names for the specified prior path.
"""
function param_perturbations(prior_path::String; n_samples::Int=10_000, quantiles = 0.1:0.1:0.9)
    # Load prior from file
    prior = CAL.get_prior(prior_path)
    
    # Generate and transform samples to physical space 
    samples = EKP.transform_unconstrained_to_constrained(
        prior, 
        EKP.sample(prior, n_samples)
    )
    param_names = get_flat_names(prior)

    # compute quantiles
    quantile_mat = Matrix(undef, length(param_names), length(quantiles))
    for (i, p) in enumerate(param_names)
        quantile_mat[i, :] = quantile(samples[i, :], quantiles)
    end
    
    return quantile_mat, param_names
end

"""
    get_all_cfsites(coszen_data_fpath::String = "/central/groups/esm/jschmitt/experiments/AtmosLossDesign/coszen_data.nc")
    
Returns a list of latitudes, longitudes, and relaxation toml files for all the cfsites.
"""
function get_all_cfsites(coszen_data_fpath::String = "/central/groups/esm/jschmitt/experiments/AtmosLossDesign/coszen_data.nc",
                         experiment_config::Dict = YAML.load_file("experiment_config.yml"))
    deep_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)
    # shallow_sites = setdiff(collect(1:119), deep_sites) # e.g., all other sites are shallow
    ds = NCDataset(coszen_data_fpath)
    lats, lons, relaxation_tomls = [], [], []
    for site in 1:119
        push!(lats, ds["lat"][site])
        push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
        if site in deep_sites
            push!(relaxation_tomls, experiment_config["forcing_toml_files"]["deep"])
        else
            push!(relaxation_tomls, experiment_config["forcing_toml_files"]["shallow"])
        end
    end
    return lats, lons, relaxation_tomls
end
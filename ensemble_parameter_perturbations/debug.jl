
function test_process_member_site(member::String, site::String, param_stats_df::DataFrame, config::Dict)
    rows = []
    
    @info "Testing member: $member, site: $site"
    
    # Get normalized parameters for this member
    norm_params = normalize_parameters("$(config["output_dir"])/$member/parameter.toml", param_stats_df)
    @info "Normalized parameters loaded successfully"
    
    data_vars = vcat(config["var_names_int"], config["var_names_prof"])
    @info "Processing variables: $data_vars"
    
    site_data_path = joinpath(config["output_dir"], member, site, "output_active")
    @info "Site data path: $site_data_path"
    
    sim_dir = SimDir(site_data_path)
    @info "SimDir created successfully"
    
    # get the type of convection from the .yml file
    forcing_type = YAML.load_file(joinpath(site_data_path, ".yml"))
    @info "YAML file loaded: $(forcing_type["toml"])"
    
    # Determine z_levels based on whether "deep" or "shallow" appears in TOML files
    toml_files = forcing_type["toml"]
    toml_string = join(toml_files, " ")
    if occursin("deep", toml_string)
        z_levels = collect(100:200:10000)  # z_levels_deep
        @info "Detected DEEP forcing - using z_levels: $(z_levels[1:5])..."
    elseif occursin("shallow", toml_string)
        z_levels = collect(100:100:4000)   # z_levels_shallow
        @info "Detected SHALLOW forcing - using z_levels: $(z_levels[1:5])..."
    else
        @error "No deep or shallow forcing found in $site_data_path. TOML string: $toml_string"
        return DataFrame()
    end
    
    try
        for data_var in data_vars
            @info "Processing variable: $data_var"
            data = get(sim_dir; short_name = data_var, reduction = "inst")
            @info "Data loaded - time range: $(data.dims["time"][1]) to $(data.dims["time"][end])"
            
            if data.dims["time"][end] < config["reduction_end_time"]
                @warn "Simulation time ($(data.dims["time"][end])) < reduction_end_time ($(config["reduction_end_time"]))"
                throw(ErrorException("Simulation time too short"))
            else
                @info "Processing time window: $(config["reduction_start_time"]) to $(config["reduction_end_time"])"
                profile_data = window(data, "time";
                                    left=config["reduction_start_time"],
                                    right=config["reduction_end_time"])
                averaged_profile = average_time(slice(profile_data, x=0, y=0))

                if !haskey(averaged_profile.dims, "z")
                    @info "No z dimension - processing as integrated variable"
                    stat = averaged_profile.data[1]
                    push!(rows, (
                        member = member,
                        site = site,
                        variable = data_var,
                        statistic = stat,
                    ))
                else
                    @info "Has z dimension - processing $(length(z_levels)) levels"
                    for zlev in z_levels
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
    catch e
        @error "Simulation failed for $member and $site: $e"
        push!(rows, (
            member = member,
            site = site,
            variable = "ERROR",
            statistic = NaN,
        ))
    end

    # Convert rows array to DataFrame and return
    result_df = DataFrame(rows)
    @info "Completed processing - $(nrow(result_df)) rows generated"
    return result_df
end
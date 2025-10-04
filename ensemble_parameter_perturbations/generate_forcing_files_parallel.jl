# Generates forcing files in parallel using Distributed.jl

using Distributed, ClusterManagers

# Add worker processes (will be managed by SLURM)
# if nprocs() == 1
#     # Add workers based on SLURM_NTASKS or default to 40
#     n_workers = haskey(ENV, "SLURM_NTASKS") ? parse(Int, ENV["SLURM_NTASKS"]) - 1 : 60
#     addprocs(n_workers)
# end
addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))

@everywhere begin
    import ClimaAtmos as CA
    using ClimaUtilities.ClimaArtifacts
    using NCDatasets
end

# Read sites from NetCDF file
ds = NCDataset("../coszen_data.nc")

lats, lons = [], []
for site in 1:119
    push!(lats, ds["lat"][site])
    push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
end

start_dates = ["20070401", "20070701", "20071001"]

# Create all combinations of work to be done
work_items = []
for start_date in start_dates
    for i in 1:lastindex(lats)
        push!(work_items, (start_date, lats[i], lons[i], i))
    end
end

println("Total work items: $(length(work_items))")
println("Number of workers: $(nworkers())")

# Function to process a single work item
@everywhere function process_forcing_file(work_item)
    start_date, lat, lon, site_index = work_item
    
    try
        single_parsed_args = Dict(
            "start_date" => start_date,
            "site_latitude" => lat,
            "site_longitude" => lon,
            "era5_diurnal_warming" => 4,
        )
        
        # Get the forcing file path 
        forcing_file_path = CA.get_external_monthly_forcing_file_path(single_parsed_args)
        
        # Log progress
        worker_id = myid()
        println("Worker $worker_id processing site $site_index (lat=$lat, lon=$lon, date=$start_date)")
        
        # Check if file exists and passes time check before generating
        if !isfile(forcing_file_path) || !CA.check_monthly_forcing_times(forcing_file_path, single_parsed_args)
            CA.generate_external_forcing_file(single_parsed_args, forcing_file_path, Float32; 
                input_data_dir = joinpath(@clima_artifact("era5_hourly_atmos_raw"), "monthly"),
                data_strs = [
                    "monthly_diurnal_profiles",
                    "monthly_diurnal_inst",
                    "monthly_diurnal_accum",
                ])
            println("Worker $worker_id completed site $site_index")
        else
            println("Worker $worker_id: forcing file already exists for site $site_index")
        end
        
        return (true, site_index, start_date, nothing)
        
    catch e
        error_msg = "Error generating forcing file for site $site_index (lat=$lat, lon=$lon, date=$start_date): $e"
        println("Worker $(myid()): $error_msg")
        return (false, site_index, start_date, error_msg)
    end
end

# Process all work items in parallel
println("Starting parallel processing...")
results = pmap(process_forcing_file, work_items)

# Summarize results
successful = sum(r[1] for r in results)
failed = length(results) - successful

println("\n" * "="^50)
println("SUMMARY:")
println("Total jobs: $(length(results))")
println("Successful: $successful")
println("Failed: $failed")

if failed > 0
    println("\nFailed jobs:")
    for result in results
        if !result[1]  # if not successful
            println("  Site $(result[2]), Date $(result[3]): $(result[4])")
        end
    end
end

println("Done generating forcing files")
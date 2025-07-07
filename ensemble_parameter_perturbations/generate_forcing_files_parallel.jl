# Generates forcing files in parallel using Distributed.jl

using Distributed

# Add worker processes (will be managed by SLURM)
if nprocs() == 1
    # Add workers based on SLURM_NTASKS or default to 40
    n_workers = haskey(ENV, "SLURM_NTASKS") ? parse(Int, ENV["SLURM_NTASKS"]) - 1 : 39
    addprocs(n_workers)
end

@everywhere begin
    import ClimaAtmos as CA
    using ClimaUtilities.ClimaArtifacts 
end

# define the sites and dates we want to use
lats = [
    -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -18.5, -17.0,
    -15.5, -14.0, -12.5, -11.0, -9.5, -8.0, 35.0, 32.0,
    29.0, 23.0, 20.0, 17.0
]

lons = [
    -72.5, -75.0, -77.5, -80.0, -82.5, -85.0, -90.0, -95.0,
    -100.0, -105.0, -110.0, -115.0, -120.0, -125.1000061,
    -125.0, -129.0, -133.0, -141.0, -145.0, -149.0
]
start_dates = ["20070101", "20070401", "20070701", "20071001"]

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
        )
        
        # get the forcing file path 
        forcing_file_path = CA.get_external_monthly_forcing_file_path(single_parsed_args)
        
        # Log progress
        worker_id = myid()
        println("Worker $worker_id processing site $site_index (lat=$lat, lon=$lon, date=$start_date)")
        println("Worker $worker_id: Forcing file path: $forcing_file_path")
        
        # generate monthly forcing file for this site
        CA.generate_external_forcing_file(single_parsed_args, forcing_file_path, Float32; 
            data_dir = joinpath(@clima_artifact("era5_hourly_atmos_raw"), "monthly"),
            data_strs = [
                "monthly_diurnal_profiles",
                "monthly_diurnal_inst",
                "monthly_diurnal_accum",
            ])
        
        println("Worker $worker_id completed site $site_index")
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
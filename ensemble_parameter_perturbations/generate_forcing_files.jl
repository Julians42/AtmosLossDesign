# Generates forcing files serially to avoid race conditions in the calibration pipeline.

import ClimaAtmos as CA
using ClimaUtilities.ClimaArtifacts 

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


for start_date in start_dates
    for i in 1:lastindex(lats)
        lat = lats[i]
        lon = lons[i]
        single_parsed_args = Dict(
            "start_date" => start_date,
            "site_latitude" => lat,
            "site_longitude" => lon,
        )
        # get the forcing file path 
        forcing_file_path = CA.get_external_monthly_forcing_file_path(single_parsed_args)
        println("Forcing file path: $forcing_file_path")
        # generate monthly forcing file for this site
        if !isfile(forcing_file_path) || !CA.check_monthly_forcing_times(forcing_file_path, single_parsed_args)
            CA.generate_external_forcing_file(single_parsed_args, forcing_file_path, Float32; 
            data_dir = joinpath(@clima_artifact("era5_hourly_atmos_raw"), "monthly"),
            data_strs = [
                "monthly_diurnal_profiles",
                "monthly_diurnal_inst",
                "monthly_diurnal_accum",
            ])
        end
        # catch e
        #     println("Error generating forcing file for site $i: $e")
        # end
    end
end
println("Done generating forcing files")

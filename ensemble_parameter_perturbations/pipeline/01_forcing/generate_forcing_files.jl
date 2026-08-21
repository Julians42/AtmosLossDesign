# Generates forcing files serially to avoid race conditions in the calibration pipeline.

import ClimaAtmos as CA
using ClimaUtilities.ClimaArtifacts
using NCDatasets

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

ds = NCDataset(joinpath(dirname(PROJECT_ROOT), "coszen_data.nc"))

lats, lons = [], []
for site in 1:119
    push!(lats, ds["lat"][site])
    push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
end

start_dates = ["20070101", "20070401", "20070701", "20071001"]

for start_date in start_dates
    for i in 1:lastindex(lats)
        lat = lats[i]
        lon = lons[i]
        single_parsed_args = Dict(
            "start_date" => start_date,
            "site_latitude" => lat,
            "site_longitude" => lon,
            "era5_diurnal_warming" => 4,
        )
        # get the forcing file path
        forcing_file_path = CA.get_external_monthly_forcing_file_path(single_parsed_args)
        println("Forcing file path: $forcing_file_path")
        # generate monthly forcing file for this site
        if !isfile(forcing_file_path) || !CA.check_monthly_forcing_times(forcing_file_path, single_parsed_args)
            CA.generate_external_forcing_file(single_parsed_args, forcing_file_path, Float32;
            input_data_dir = joinpath(@clima_artifact("era5_hourly_atmos_raw"), "monthly"),
            data_strs = [
                "monthly_diurnal_profiles",
                "monthly_diurnal_inst",
                "monthly_diurnal_accum",
            ])
        end
    end
end
println("Done generating forcing files")

using Revise 
using Infiltrator
import YAML
import ClimaAtmos as CA
using Dates
using NCDatasets
using CairoMakie



parsed_args = YAML.load_file("config/model_configs/prognostic_edmfx_tv_era5driven_column.yml")
forcing_file_path = CA.get_external_forcing_file_path(parsed_args)
CA.generate_multiday_external_forcing_file(parsed_args, forcing_file_path, Float64)


config = CA.AtmosConfig("config/model_configs/prognostic_edmfx_tv_era5driven_column.yml")
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)
println("Done!")



# open a new file, check out the data keys that it has 

start_date = DateTime(parsed_args["start_date"], "yyyymmdd")
end_time = start_date + Dates.Second(CA.time_to_seconds(parsed_args["t_end"]))
start_dates = start_date:Day(1):end_time

file_list = String[]
for (i, dd) in enumerate(start_dates)
    single_parsed_args = Dict(
        "start_date" => Dates.format(dd, "yyyymmdd"),
        "t_end" => "23hours", # some value between 0 and 24 hours to generate the single file
        "site_latitude" => parsed_args["site_latitude"],
        "site_longitude" => parsed_args["site_longitude"],
    )
    single_file_path = CA.get_external_forcing_file_path(single_parsed_args)
    push!(file_list, single_file_path)
end

concat_ds = NCDataset(forcing_file_path, "c")

day_ds = NCDataset(file_list[1], "r")

ds = Dataset(file_list; aggdim="time")




# visualize smoothing effects
var = "w"
fig = Figure(size = (800, 600))

ax1 = Axis(fig[1, 1], title = "Unsmoothed")
heatmap!(ax1, ds[var][100, 100, :, :])

ax2 = Axis(fig[1, 2], title = "Smoothed")
heatmap!(ax2, CA.smooth_4D_era5(ds, var, 100, 100))

ax3 = Axis(fig[1, 3], title = "diff")
heatmap!(ax3, ds[var][100, 100, :, :] .- CA.smooth_4D_era5(ds, var, 100, 100))
save("test.png", fig)


var = "q"
fig = Figure(size = (1200, 400))

# Flip along pressure_level (dim 1)
unsmoothed = transpose(ds[var][100, 100, :, :])
smoothed = transpose(CA.smooth_4D_era5(ds, var, 100, 100))
diff = unsmoothed .- smoothed

# Unsmoothed plot + colorbar
ax1 = Axis(fig[1, 1], title = "Unsmoothed")
hm1 = heatmap!(ax1, unsmoothed)
Colorbar(fig[1, 2], hm1)

# Smoothed plot + colorbar
ax2 = Axis(fig[1, 3], title = "Smoothed")
hm2 = heatmap!(ax2, smoothed)
Colorbar(fig[1, 4], hm2)

# Diff plot + colorbar
ax3 = Axis(fig[1, 5], title = "Diff")
hm3 = heatmap!(ax3, diff, colormap=:balance)
Colorbar(fig[1, 6], hm3)

save("test.png", fig)


# check the 3D version
var = "skt"
fig = Figure(size = (1200, 400))

# Flip along pressure_level (dim 1)
unsmoothed = ds3[var][100, 100, :]
smoothed = CA.smooth_3D_era5(ds3, var, 100, 100)
diff = unsmoothed .- smoothed

# Unsmoothed plot + colorbar
ax1 = Axis(fig[1, 1], title = "Unsmoothed")
hm1 = lines!(ax1, unsmoothed)

# Smoothed plot + colorbar
ax2 = Axis(fig[1, 3], title = "Smoothed")
hm2 = lines!(ax2, smoothed)

# Diff plot + colorbar
ax3 = Axis(fig[1, 5], title = "Diff")
hm3 = lines!(ax3, diff)

save("test_3d.png", fig)
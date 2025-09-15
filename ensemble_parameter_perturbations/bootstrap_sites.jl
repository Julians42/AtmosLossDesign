# bootstrap the sites 
import EnsembleKalmanProcesses as EKP
import YAML
import ClimaCalibrate as CAL
using Statistics
using CairoMakie
using DataFrames
using DataFramesMeta
import CSV
using LinearAlgebra
using FixedEffectModels
import TOML
using Glob
using JLD2

include("new_helper_funcs.jl")

# load the config
config = YAML.load_file("experiment_config.yml")

# read in all 100 dataframes into one concatenated dataframe
df = @. CSV.read(glob("postprocessing/dataframes/diagnostic_edmfx/*.csv"), DataFrame) 

df = postprocess_dataframe(df) # filters nans and normalizes statistics 

# bootstrap the sites
n_bootstraps = 100

for i in 1:n_bootstraps
    # sample the sites with replacement
    unique_sites = unique(df.site)
    sites = Set(sample(unique_sites, length(unique_sites), replace=true))
    
    # create a new dataframe with the sampled sites
    df_sampled = df[in.(df.site, Ref(sites)), :]

    # compute the information gain
    all_variables = get_all_variables(config, Config_cfsites_deep())
    full_ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering =
        information_gain(all_variables, df_sampled, Linear())

    # define bootstrap directory
    outdir = joinpath("bootstrap_sites", "bootstrap_$i")
    mkpath(outdir)  # ensure directory exists

    # save everything into one .jld2 file
    outfile = joinpath(outdir, "results.jld2")
    @save outfile full_ig ∇G Σ_y Σ_0 constrained_params param_ordering
end
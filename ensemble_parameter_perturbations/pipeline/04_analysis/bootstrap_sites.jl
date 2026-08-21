# Resamples sites with replacement to quantify bootstrap uncertainty in the
# information-gain analysis. See pipeline/04_analysis/point_estimate.jl for
# the non-bootstrapped point estimate these samples are compared against.
import EnsembleKalmanProcesses as EKP
import YAML
import ClimaCalibrate as CAL
using Statistics
using DataFrames
using DataFramesMeta
import CSV
using LinearAlgebra
using FixedEffectModels
import TOML
using Glob
using JLD2

PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "methods.jl"))
include(joinpath(PROJECT_ROOT, "src", "parameter_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiment.jl"))

# load the config
config = YAML.load_file(joinpath(PROJECT_ROOT, "config", "experiment_config.yml"))
resolve_config_paths!(config, PROJECT_ROOT)

# read in all 100 per-member dataframes into one concatenated dataframe
df = @. CSV.read(glob(joinpath(PROJECT_ROOT, "data", "postprocessing", "diagnostic_edmfx", "*.csv")), DataFrame)

df = normalize_statistics(df) # filters nans and normalizes statistics

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
        information_gain(all_variables, df_sampled, Linear(); rootdir = config["output_dir"], prior_path = config["prior_path"])

    # define bootstrap directory
    outdir = joinpath(PROJECT_ROOT, "data", "bootstrap_sites", "bootstrap_$i")
    mkpath(outdir)  # ensure directory exists

    # save everything into one .jld2 file
    outfile = joinpath(outdir, "results.jld2")
    @save outfile full_ig ∇G Σ_y Σ_0 constrained_params param_ordering
end

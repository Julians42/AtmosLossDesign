# Computes the non-bootstrapped ("full sample") information-gain point
# estimate and saves it to data/bootstrap_sites/no_bootstrap/results.jld2.
#
# This is the point estimate that pipeline/05_plots/marginals.jl,
# resolution.jl, and experiment_scenarios.jl all load, alongside the
# bootstrap replicates from bootstrap_sites.jl for uncertainty bands.
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

all_variables = get_all_variables(config, Config_cfsites_deep())
full_ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering =
    information_gain(all_variables, df, Linear(); rootdir = config["output_dir"], prior_path = config["prior_path"])

outdir = joinpath(PROJECT_ROOT, "data", "bootstrap_sites", "no_bootstrap")
mkpath(outdir)
outfile = joinpath(outdir, "results.jld2")
@save outfile full_ig ∇G Σ_y Σ_0 constrained_params param_ordering
@info "Saved point estimate to $outfile"

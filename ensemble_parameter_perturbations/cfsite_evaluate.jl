"""
evaluate.jl

Script to evaluate regression analysis from preprocessed dataframes.
Takes cleaned dataframes and produces regression coefficient analysis plots.
"""
# Ensure Pkg is available and activate environment
# using Pkg
# cd("/net/sampo/data1/jschmitt/AtmosLossDesign/ensemble_parameter_perturbations")
# Pkg.activate(".") 

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

all_variables = get_all_variables(config, Config_cfsites_deep())
full_ig, ∇G, Σ_y, Σ_0, constrained_params, param_ordering = information_gain(all_variables, df, Linear())

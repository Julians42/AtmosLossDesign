import EnsembleKalmanProcesses as EKP
import ScikitLearn
import YAML
import JLD2
import ClimaCalibrate as CAL
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis
using Revise
using DataFrames
import CSV
using ColorSchemes
using FixedEffectModels
using TOML
using LinearAlgebra

include("var_helper_funcs.jl")

# include(joinpath(@__DIR__, "..", "helper_funcs.jl"))
config = YAML.load_file("experiment_config.yml");

df = get_observations(config)
CSV.write("grad_q_data.csv", df)
println("Completed")

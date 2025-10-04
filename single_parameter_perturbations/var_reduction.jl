# compute the reduction in variance associated with observing a set of observations 

import YAML
import JLD2
using Statistics
using CairoMakie
using Glob
using ClimaAnalysis
using DataFrames
using CSV
using NaNStatistics
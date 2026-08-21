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
import CalibrateEmulateSample as CES
using FixedEffectModels
using LinearAlgebra

"""
    extract_parameter_stats(prior)

Extract mean and standard deviation statistics from parameter distributions.
Returns a DataFrame with parameter names, means, standard deviations, and types.
"""
function extract_parameter_stats(prior)
    prior_rows = []

    for (i, param_dist) in enumerate(prior.distribution)
        param_name = prior.name[i]
        if param_dist isa EKP.Parameterized
            # For scalar parameters
            push!(prior_rows, (
                param_name,
                param_dist.distribution.μ,  # mean
                param_dist.distribution.σ,  # standard deviation
                "scalar"
            ))
        elseif param_dist isa EKP.VectorOfParameterized
            # For vector parameters
            for (j, dist) in enumerate(param_dist.distribution)
                push!(prior_rows, (
                    "$(param_name)[$j]",
                    dist.μ,  # mean
                    dist.σ,  # standard deviation
                    "vector"
                ))
            end
        end
    end

    # Create DataFrame with parameter statistics
    return DataFrame(
        param = [row[1] for row in prior_rows],
        mean = [row[2] for row in prior_rows],
        std = [row[3] for row in prior_rows],
        type = [row[4] for row in prior_rows]
    )
end

"""
    normalize_parameters(θ, normalization_params)

Normalize parameters using the mean and standard deviation from the prior distributions.
θ should be a dictionary or named tuple of parameters.
"""
function normalize_parameters(θ, normalization_params)
    normalized_θ = Dict{String, Float64}()

    for row in eachrow(normalization_params)
        param_name = row.param
        if endswith(param_name, "]")  # Vector parameter
            base_name = param_name[1:findfirst('[', param_name)-1]
            idx = parse(Int, param_name[findfirst('[', param_name)+1:end-1])
            if haskey(θ, base_name)
                normalized_θ[param_name] = (θ[base_name][idx] - row.mean) / row.std
            end
        else  # Scalar parameter
            if haskey(θ, param_name)
                normalized_θ[param_name] = (θ[param_name] - row.mean) / row.std
            end
        end
    end

    return normalized_θ
end

"""
    main()

Main function to run the parameter normalization process.
"""
function main()
    # Get the prior distributions
    prior = CAL.get_prior(joinpath(@__DIR__, "prior.toml"))

    # Extract and save parameter statistics
    normalization_params = extract_parameter_stats(prior)
    JLD2.save(joinpath(@__DIR__, "parameter_normalization.jld2"), "normalization_params", normalization_params)

    println("Parameter normalization statistics:")
    display(normalization_params)

    return normalization_params
end

# Run the main function if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

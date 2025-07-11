using LinearAlgebra
using DataFrames

#=
----------------------------------------------------------------------
STEP 1: DEFINE THE METRIC TYPES
----------------------------------------------------------------------
We define an abstract type `InformationMetric` to represent any kind
of information calculation. Then, we create specific, concrete subtypes
for each method we want to implement. This allows Julia's dispatch
system to select the correct computation at runtime.
=#

"""
Abstract supertype for all information content metrics.
"""
abstract type InformationMetric end

"""
    TraceMetric()

A metric that computes the trace of the parameter-space information matrix
(J' * J). This is equivalent to the sum of squared Frobenius norms of the
Jacobian rows. It measures total sensitivity but does not account for
redundancy between observations.
"""
struct TraceMetric <: InformationMetric end

"""
    EffectiveRankMetric()

A metric that computes the effective rank of the Jacobian sub-matrix (J_sub).
This measures the number of "significant" dimensions of information,
providing a value between 1 and min(k, d) where k is the number of
observations and d is the number of parameters. It naturally handles
redundancy and reveals saturation.
"""
struct EffectiveRankMetric <: InformationMetric end

"""
    LogDeterminantMetric(epsilon::Float64 = 1e-10)

A metric that computes the sum of the logarithms of the eigenvalues of the
observation-space information matrix (J * J'). It quantifies the volume of
the information space and is sensitive to redundancy. An epsilon is added
for numerical stability to handle near-zero eigenvalues.
"""
struct LogDeterminantMetric <: InformationMetric
    epsilon::Float64
end
# Provide a default constructor if no epsilon is given
LogDeterminantMetric() = LogDeterminantMetric(1e-10)


#=
----------------------------------------------------------------------
STEP 2: IMPLEMENT THE COMPUTATION FOR EACH METRIC
----------------------------------------------------------------------
Here, we create a function `compute_metric` with multiple methods.
Each method is specialized for one of our concrete metric types.
Julia will automatically call the correct one based on the type of
the `metric` argument passed to it.
=#

"""
    compute_metric(J_sub, metric::TraceMetric)

Calculates the information content using the trace method.
`J_sub` is the Jacobian sub-matrix for a specific variable.
"""
function compute_metric(J_sub::AbstractMatrix, ::TraceMetric)
    # J_sub' * J_sub is the d x d parameter-space information matrix
    # The trace is the sum of its eigenvalues, representing total variance.
    return tr(J_sub' * J_sub)
end

"""
    compute_metric(J_sub, metric::EffectiveRankMetric)

Calculates the information content using the effective rank method.
"""
function compute_metric(J_sub::AbstractMatrix, ::EffectiveRankMetric)
    # Don't compute the full SVD, just the singular values for efficiency
    sv = svdvals(J_sub)
    # Avoid division by zero if all singular values are zero
    sum_sv_sq = sum(sv .^ 2)
    return sum_sv_sq > 0 ? (sum(sv)^2) / sum_sv_sq : 0.0
end

"""
    compute_metric(J_sub, metric::LogDeterminantMetric)

Calculates the information content using the log-determinant method.
"""
function compute_metric(J_sub::AbstractMatrix, metric::LogDeterminantMetric)
    # J_sub * J_sub' is the k x k observation-space information matrix
    # Adding epsilon * I ensures the matrix is positive definite for the log.
    k = size(J_sub, 1)
    H_y_sub = J_sub * J_sub' + metric.epsilon * I(k)
    return logdet(H_y_sub)
end


#=
----------------------------------------------------------------------
STEP 3: REFACTOR THE MAIN ANALYSIS FUNCTION
----------------------------------------------------------------------
This new function, `analyze_variable_information`, replaces your
original function. It takes a `metric` object as an argument and
uses it to dispatch to the correct computation.
=#

"""
    analyze_variable_information(reg_coefs, informing_variables, var_names_prof, var_names_int, exp_name, metric::InformationMetric)

Analyzes the information content of different variables using a specified metric.

# Arguments
- `reg_coefs`: The full regression coefficient matrix (J*), with dimensions (d x m).
- `informing_variables`: A vector of strings identifying each observation (column of reg_coefs).
- `var_names_prof`: A vector of strings for the profile variable names (e.g., "ta", "hus").
- `var_names_int`: A vector of strings for the integrated variable names (e.g., "pr", "lwp").
- `exp_name`: A string for the experiment name, used as the column header in the output DataFrame.
- `metric`: An object of a type inheriting from `InformationMetric` (e.g., `TraceMetric()`, `EffectiveRankMetric()`).

# Returns
- A `DataFrame` with columns `:var_name` and a column named after `exp_name` containing the calculated information content.
"""
function analyze_variable_information(
    reg_coefs::AbstractMatrix,
    informing_variables::Vector,
    var_names_prof::Vector,
    var_names_int::Vector,
    exp_name::String,
    metric::InformationMetric
)
    var_informing_results = []

    # --- Process Profile Variables ---
    for profile_var in var_names_prof
        # Find all columns corresponding to this profile variable
        matching_indices = findall(v -> startswith(v, profile_var), informing_variables)

        if isempty(matching_indices)
            println("Warning: No matching observations found for profile variable: ", profile_var)
            continue
        end

        # Extract the relevant sub-matrix of the Jacobian (J_sub)
        # Note: reg_coefs is d x m, so we need to select columns. The result is d x k.
        # We then transpose it to get the k x d matrix convention used in the math.
        J_sub = permutedims(reg_coefs[:, matching_indices]) # k x d matrix

        # Dispatch to the correct metric computation
        info_value = compute_metric(J_sub, metric)
        println("$(profile_var): $(info_value) (using $(typeof(metric)))")
        push!(var_informing_results, [profile_var, info_value])
    end

    # --- Process Integrated Variables ---
    for single_var in var_names_int
        matching_indices = findall(v -> startswith(v, single_var), informing_variables)

        if isempty(matching_indices)
            println("Warning: No matching observations found for integrated variable: ", single_var)
            continue
        end
        
        if length(matching_indices) > 1
             println("Warning: Multiple observations found for integrated variable: $(single_var). Treating them as a combined profile.")
        end

        # This is essentially a profile with one or few levels
        J_sub = permutedims(reg_coefs[:, matching_indices])

        # Dispatch to the correct metric computation
        info_value = compute_metric(J_sub, metric)
        println("$(single_var): $(info_value) (using $(typeof(metric)))")
        push!(var_informing_results, [single_var, info_value])
    end

    if isempty(var_informing_results)
        return DataFrame(:var_name => String[], Symbol(exp_name) => Float64[])
    end

    # Convert the array of results to a DataFrame
    var_informing_matrix = hcat(var_informing_results...)
    var_informing_df = DataFrame([
        :var_name => var_informing_matrix[1, :],
        Symbol(exp_name) => var_informing_matrix[2, :]
    ])

    return var_informing_df
end


#=
----------------------------------------------------------------------
STEP 4: EXAMPLE USAGE
----------------------------------------------------------------------
Here's how you would use the new, flexible function.
=#

function main()
    # --- Mock Data Setup ---
    # Imagine we have 5 parameters (d=5) and 23 observations (m=23)
    d = 5
    m = 23
    reg_coefs = randn(d, m) # d x m Jacobian (J*')

    # Observations: 10 levels of 'ta', 10 of 'hus', and 3 integrated vars
    informing_variables = [
        ["ta_$(i)" for i in 1:10]...,
        ["hus_$(i)" for i in 1:10]...,
        "pr", "lwp", "clt"
    ]
    var_names_prof = ["ta", "hus"]
    var_names_int = ["pr", "lwp", "clt"]
    exp_name = "Experiment1"

    println("--- 1. Using the Trace Metric (Your Original Method) ---")
    df_trace = analyze_variable_information(
        reg_coefs,
        informing_variables,
        var_names_prof,
        var_names_int,
        exp_name,
        TraceMetric()
    )
    println(df_trace)
    println("\n" * "="^50 * "\n")


    println("--- 2. Using the Effective Rank Metric (Recommended for Saturation) ---")
    df_eff_rank = analyze_variable_information(
        reg_coefs,
        informing_variables,
        var_names_prof,
        var_names_int,
        exp_name,
        EffectiveRankMetric()
    )
    println(df_eff_rank)
    println("\n" * "="^50 * "\n")


    println("--- 3. Using the Log-Determinant Metric ---")
    df_logdet = analyze_variable_information(
        reg_coefs,
        informing_variables,
        var_names_prof,
        var_names_int,
        exp_name,
        LogDeterminantMetric()
    )
    println(df_logdet)
    println("\n" * "="^50 * "\n")
end

# Run the example
main()

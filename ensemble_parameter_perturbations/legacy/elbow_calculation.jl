function dimension_by_kneedle(s::AbstractVector{<:Real})
    n = length(s)
    if n <= 2
        return 1
    end

    # Normalize both x and y to [0,1]
    x = collect(1:n)
    x_norm = (x .- 1) ./ (n - 1)  # Normalize x to [0,1]
    y_norm = (s .- minimum(s)) ./ (maximum(s) - minimum(s))  # Normalize y to [0,1]

    # For scree plots, we expect decreasing values, so flip y_norm
    y_norm = 1 .- y_norm

    # Compute differences from the straight line connecting first and last points
    # Line equation: y = mx + b where m = (y2-y1)/(x2-x1), b = y1
    x1, x2 = x_norm[1], x_norm[end]
    y1, y2 = y_norm[1], y_norm[end]

    # Calculate perpendicular distance from each point to the line
    distances = Float64[]
    for i in 1:n
        # Distance from point (x_norm[i], y_norm[i]) to line connecting endpoints
        # Using point-to-line distance formula
        distance = abs((y2 - y1) * x_norm[i] - (x2 - x1) * y_norm[i] + x2 * y1 - y2 * x1) /
                  sqrt((y2 - y1)^2 + (x2 - x1)^2)
        push!(distances, distance)
    end

    # Return the index with maximum distance (the "elbow")
    return argmax(distances)
end

# Alternative elbow detection using second derivative
function elbow_second_derivative(s::AbstractVector{<:Real})
    n = length(s)
    if n <= 3
        return 1
    end

    # Calculate second derivatives
    second_derivs = Float64[]
    for i in 2:(n-1)
        second_deriv = s[i-1] - 2*s[i] + s[i+1]
        push!(second_derivs, second_deriv)
    end

    # Find the point with maximum second derivative (most curvature)
    return argmax(second_derivs) + 1  # +1 because we start from index 2
end

# Alternative: Simple percentage-based cutoff
function elbow_percentage_cutoff(s::AbstractVector{<:Real}, threshold::Float64 = 0.05)
    n = length(s)
    if n <= 1
        return 1
    end

    total_variance = sum(s)
    cumulative_variance = cumsum(s)
    cumulative_percentage = cumulative_variance ./ total_variance

    # Find first point where we've captured (1-threshold) of total variance
    cutoff_idx = findfirst(x -> x >= (1 - threshold), cumulative_percentage)
    return cutoff_idx === nothing ? n : cutoff_idx
end

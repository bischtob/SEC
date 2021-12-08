# TODO: DESIGN PROPOSALS

"""
torus(x, a, b)
# Description
- Takes x ∈ ℝ and outputs torus(x) ∈ [a, b] in a periodic way.
- If a particle is moving to the right then it will pop from b to the point a
# Arguments: x, a, b
- `x`: (scalar). Current location of particle
- `a`: (scalar). left endpoint of interval
- `b`: (scalar). right endpoint of interval
# Output
-  `y`: (scalar). a value in the interval [a,b]
"""
torus(x::Number, a::Number, b::Number) = (((x-a)/(b-a))%1 - 0.5 * (sign((x-a)/(b-a)) - 1) )*(b-a) + a

"""
torus(x, a, b)
# Description
- Takes x ∈ ℝⁿ and outputs torus(x) ∈ ∏[aⁿ, bⁿ] in a periodic way.
- If a particle is moving to the right then it will pop from one part of the box to the oher
# Arguments: x, a, b
- `x`: (array). Current location of particle
- `a`: (array). left endpoint of tensor product interval
- `b`: (array). right endpoint of tensor product interval
# Output
-  `y`: (array). a value in the interval ∏[aⁿ, bⁿ]
"""
function torus(x::AbstractArray, a::AbstractArray, b::AbstractArray)
    N = length(x)
    y = zeros(N)
    for i in 1:N
        y[i] = torus(x[i], a[i], b[i])
    end
    return y
end

"""
closure_proprosal(covariance = Σ; left_bounds = [], right_bounds = []))
# Description
- Constructs a proposal for the Monte Carlo method.
# Arguments
- `covariance`: (vector) proposal parameter
# Keyword Arguments
- `left_bounds`: (array), left bounds for parameters
- `right_bounds`: (array), right bounds for parameters
# Output:
- `proposal`: (function), a function that outputs the proposal parameter
"""
function closure_proposal(Σ; left_bounds = [], right_bounds = [])
    perturbation = MvNormal(Σ)
    function proposal(C)
        proposal_C = C + rand(perturbation)
        # limit ranges for the parameters
        if isempty(left_bounds)
            return proposal_C
        else
            return torus(proposal_𝑪, left_bounds, right_bounds)
        end
        return proposal_C
    end
    return proposal
end
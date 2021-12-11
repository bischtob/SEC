"""
GaussianProcess{M, C}


# Restrictions
must be able to call mean(x) and covariance(x,y)
"""
struct GaussianProcess{M, C}
    mean::M
    covariance::C
end

function mean(gp::GaussianProcess{M,C}, X) where {M,C}
    return [gp.mean(X[i]) for i in eachindex(X)]
end

function cov(gp::GaussianProcess{M,C}, X) where {M,C}
    return [gp.covariance(X[i], X[j]) for i in eachindex(X), j in eachindex(X)]
end

function cond_cov(gp::GaussianProcess{M,C}, X, Xc) where {M,C} end

function predict end
function uncertainty end
function rand end

function nll(gp::GaussianProcess{M,C}, X, Y)
    K = cov(gp, X)
    n = length(X)

    return -0.5*Y'* (K \ Y) - 0.5*log(det(K))
end

# example
covariance = ExponentiatedSquare(σ = 2.0) # Similar do Dense(W,b)
gp =  GaussianProcess(x -> 0.0, (x,y) -> covariance(x,y))

loss(X,Y) = nll(gp, X, Y)

#=
newGP = condition(GP, data)
struct ConditionalMean{S}
end
struct ConditionalCovariance{S}
end
=#

export GaussianProcess

"""
GaussianProcess{M, C}

# Restrictions
must be able to call mean(x) and covariance(x,y)
"""
Base.@kwdef struct GaussianProcess{M, C}
    mean::M
    covariance::C
end

# Extend Distributions functions 
mean(gp::GaussianProcess) = gp.mean
cov(gp::GaussianProcess) = gp.covariance

function mean(gp::GaussianProcess, X)
    return [gp.mean(X[i]) for i in eachindex(X)]
end

# Statistics.jl has a mean(::Any, ::AbstractArray), thus disambiguate
function mean(gp::GaussianProcess, X::S) where {S <: AbstractArray}
    return [gp.mean(X[i]) for i in eachindex(X)]
end

function cov(gp::GaussianProcess, X)
    return [gp.covariance(X[i], X[j]) for i in eachindex(X), j in eachindex(X)]
end

function condition(gp::GaussianProcess, data)
    (; X, Y) = data

    K = cov(gp, X) 
    predictor = K \ Y

    function cond_mean(x)
        cX = [gp.covariance(x, dx) for dx in X]
        return gp.mean(x) + predictor' * (cX .- gp.mean(x))
    end

    function cond_cov(x,y)
        cX = [gp.covariance(x, dx) for dx in X]
        cY = [gp.covariance(y, dx) for dx in X]
        return gp.covariance(x, y) - cX' * (K \ cY)
    end

    return GaussianProcess(mean = cond_mean, covariance = cond_cov)
end

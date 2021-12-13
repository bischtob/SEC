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
mean(gp::GaussianProcess) =  gp.mean
cov(gp::GaussianProcess) =  gp.covariance

function mean(gp::GaussianProcess, X)
    return [gp.mean(X[i]) for i in eachindex(X)]
end

# statistics has a mean(::Any, ::AbstractArray), thus disambiguate
function mean(gp::GaussianProcess, X::S) where {S <: AbstractArray}
    return [gp.mean(X[i]) for i in eachindex(X)]
end

function cov(gp::GaussianProcess, X)
    return [gp.covariance(X[i], X[j]) for i in eachindex(X), j in eachindex(X)]
end

#=
function cond_cov(gp::GaussianProcess{M,C}, X, Xc) where {M,C} end

function predict end
function uncertainty end
function rand end

function nll(gp::GaussianProcess{M,C}, X, Y)
    K = cov(gp, X)
    n = length(X)

    return -0.5*Y'* (K \ Y) - 0.5*log(det(K))
end
=#
# example
#=
covariance = ExponentiatedSquare(σ = 2.0) # Similar do Dense(W,b)
gp =  GaussianProcess(x -> 0.0, (x,y) -> covariance(x,y))

loss(X,Y) = nll(gp, X, Y)
=#
#=
newGP = condition(GP, data)
struct ConditionalMean{S}
end
struct ConditionalCovariance{S}
end

factorize(x, y::Nothing) = x
factorize(x, ::Cholesky) = cholesky(x)

# add in sparse cholesky, example on extending
SparseCholesky() = SparseCholesky(robust = true, threshold = 0.2)
function factorize(K, factorization::SparseCholesky)
        mK = maximum(K)
        (; robust, threshold) = factorization
        # make Cholesky factorization robust 
        # by adding a small amount to the diagonal
        if robust
            K += mK*sqrt(eps(1.0))*I
        end
        # check sparsity
        bools = K .> entry_threshold * mK
        sparsity = sum(bools) / length(bools)

        # use sparse cholesky if below threshold
        # otherwise return normal 
        if sparsity < sparsity_threshold
            sparse_K = similar(K) .* 0
            sparse_K[bools] = sK[bools]
            K = sparse(Symmetric(sparse_K))
            CK = cholesky(K)
        else
            CK = cholesky(K)
        end
end


function condition(gp::GaussianProcess, data::FormatA; factorization = nothing)
    (; X, Y) = data
    # factorize once since it shows up in both
    # the mean and covariance
    K = cov(gp, X)
    K = factorize(K, factorization)
    cond_cov  = ConditionalCovariance(gp, X, K)
    cond_mean = ConditionalMean(gp, X, K)
    return GaussianProcess(mean = cond_mean, covariance = cond_cov)
end
# this (we lose info about data)
function ConditionalCovariance(gp, X, K)
    function cov(x, y)
        cX = [gp.k(x, dx) for dx in X]
        cY = [gp.k(y, dx) for dx in X]
        var = gp.k(x, y) .- cX' * (K \ cY)
    end
end
# OR 
# the nice part about this way is we have a record
# of the datgp.covariance that has been seen
struct ConditionalCovariance{GP, DATA, MAT}
    gp::GP
    X::DATA
    K::MAT
end
function (cc::ConditionalCovariance)(x,y)
        cX = [cc.gp.covariance(x, dx) for dx in cc.X]
        cY = [cc.gp.covariance(y, dx) for dx in cc.X]
        var = cc.gp.covariance(x, y) - cX' * (cc.K \ cY)
end
# for 1 layer of data
# gp.covariance.gp.covariance
# for 2 layer of data
# gp.covariance.gp.covariance.gp.covariance 
# could probably define
original_covariance(covariance::ConditionalCovariance) = original_covariance(covariance.gp)
original_covariance(covariance::Function) = covariance
# jumpin through the tree is also possible
=#

##
#=
struct DispatchTest{S, T}
    a::S
    b::T
end

function dispatch(::DispatchTest)
    println("dispatch default")
end

function dispatch(::DispatchTest{S}) where {S <: Number}
    println("dispatch Number")
end

function dispatch(::DispatchTest{S}) where {S <: Float64}
    println("dispatch Float64")
end

function dispatch(::DispatchTest{S, T}) where {S <: Float64, T <: Float64}
    println("dispatch Float64, Float64")
end

dispatch(DispatchTest(nothing,nothing))
dispatch(DispatchTest(1,nothing))
dispatch(DispatchTest(1.0,nothing))
dispatch(DispatchTest(1.0,1.0))
=#
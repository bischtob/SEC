using Random, LinearAlgebra, Distributions
⊗(a,b) = a * b'
Random.seed!(1234)

# Here we go through an example where the theory "works perfectly"

# Define forward map
# input Space 
Mo = 2
# output space
M = 2
# Define forward map as a random matrix
H = randn(M,Mo) 
forward_map(x) = H * x

# Create artificial solution
x = randn(Mo) .- 10 
y̅ = forward_map(x)

# Define an initial ensemble distribution
MoMo = randn(Mo,Mo)
Σ = MoMo' * MoMo + I 
prior = MvNormal(Σ)
# this choice implies prior ∝ exp( - x' Σ⁻¹ x)

# Implicitly define the likelihood function via the covariance 
MM = randn(M,M)
Γ = MM'*MM + I
# implies likelihood  exp(- T * dot(y - H * x, Γ⁻¹ * (y - H * x)) )
# where T is the final time of the simulation
# Observe that
# likelihood = exp(- dot(y - H * x, Γ⁻¹ * (y - H * x)) ) for T = 1

# number of steps
N = 1 
# timestep size (h = 1/N will converge to posterior)
h = 1/N 
# number of ensemble members
J = 1000
# Likelihood perturbation for algorithm, (constructed from before)
ξ = MvNormal(1/h * Γ)

# Construct empirical prior
u = [rand(prior) for i in 1:J]
u₀ = copy(u)

# Viewed as a minimization problem, we are minimizing
# dot(y - H * x, Γ⁻¹ * (y - H * x)) + dot(x, Σ⁻¹ x)
# whose solution is 
# (H' Γ⁻¹ H + Σ⁻¹)⁻¹ H' Γ⁻¹ y
# Alternatively we can view the posterior as  
# posterior ∝ exp(-dot(y - H * x, Γ⁻¹ * (y - H * x)) - x' Σ⁻¹ x)
# which implies μ = (H' Γ⁻¹ H + Σ⁻¹)⁻¹ H' Γ⁻¹ y
# and           σ = H' Γ⁻¹ H + Σ⁻¹

for i in 1:N
    u̅ = mean(u)
    G = forward_map.(u) # error handling needs to go here
    G̅ = mean(G)

    # define covariances
    Cᵘᵖ = (u[1] - u̅) ⊗ (G[1] - G̅)
    Cᵖᵖ = (G[1] - G̅) ⊗ (G[1] - G̅)
    for j in 2:J
        Cᵘᵖ += (u[j] - u̅) ⊗ (G[j] - G̅)
        Cᵖᵖ += (G[j] - G̅) ⊗ (G[j] - G̅)
    end
    Cᵘᵖ *= 1/J
    Cᵖᵖ *= 1/J

    # ensemblize the data
    y = [y̅ + rand(ξ) for i in 1:J]
    r = y - G

    # update
    Cpp_inv = cholesky(Symmetric(Cᵖᵖ + 1/h * Γ))
    for j in 1:J
        u[j] += Cᵘᵖ * ( Cpp_inv \ r[j] )
    end
end

# This 
approx = mean(u)
if length(approx) < 4
    println("mean appro = ", approx)
end
# should converge to this
HΓH = Symmetric((H'* (Γ \ H) + cholesky(Σ) \ I))
cHΓH = cholesky(HΓH) # LU fails
rhs = (H'*(Γ\y̅[:]))
exact = cHΓH\rhs
if length(exact) < 4
    println("mean exact = ", exact) 
end
println("mean relative error = ", norm(approx - exact) / norm(exact))
println("loss relative error = ", norm(H*approx - H*exact) / norm(H*exact))

exact_cov = cHΓH \ I
approx_cov = cov(u)
if length(exact) < 4
    @show approx_cov
    @show exact_cov
end
println("cov relative error = ", norm(approx_cov - exact_cov) / norm(exact_cov) ) 

#=
using GLMakie
fig, ax, sc = scatter([(u[i][1], u[i][2]) for i in eachindex(u)], color = :red)

scatter!(ax, [(u₀[i][1], u₀[i][2]) for i in eachindex(u)], color = :blue)
scatter!(ax, [(exact[1], exact[2])], marker = '⋆', color = :yellow, markersize = 30)
display(fig)
=#
##
# calibrate(forward_map, [], )


##
# ξ = MvNormal(1/h * Γ)
Δt = h

"""
eki_step!(u, forward_map, y̅, J, ξ, Γ, Δt)

# Description 
Take a single step of the Ensemble Kalman Inversion (EKI) algorithm

# Arguments 
`u`: Vector{S} where S. Empirical prior distribution 
`forward_map`: Function. Operates on objects of type S
`data`: Vector or Array
"""
function eki_step!(u, forward_map, data, J, ξ, Γ, Δt)
    u̅ = mean(u)
    G = forward_map.(u) # error handling needs to go here
    G̅ = mean(G)

    Cᵘᵖ = (u[1] - u̅) ⊗ (G[1] - G̅)
    Cᵖᵖ = (G[1] - G̅) ⊗ (G[1] - G̅)
    for j in 2:J
        Cᵘᵖ += (u[j] - u̅) ⊗ (G[j] - G̅)
        Cᵖᵖ += (G[j] - G̅) ⊗ (G[j] - G̅)
    end
    Cᵘᵖ *= 1/J
    Cᵖᵖ *= 1/J

    y = [data + rand(ξ) for j in 1:J]
    residual = y - G

    factored_Cᵖᵖ = cholesky(Symmetric(Cᵖᵖ + 1/Δt * Γ))
    for j in 1:J
        u[j] += Cᵘᵖ * ( factored_Cᵖᵖ \ residual[j] )
    end 
end

function eki_loop!(u, forward_map, y̅, J, Γ, Δt, N; random_seed = 1234)
    Random.seed!(random_seed)
    ξ = MvNormal(1/Δt * Γ)
    for n in 1:N
        eki_step!(u, forward_map, y̅, J, ξ, Γ, Δt)
    end
end

##
# calibrate!(forward_map, x, y, method = EKI())
optimize!(loss, x, method = EKI())

#


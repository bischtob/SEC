using Random, LinearAlgebra, Distributions
⊗(a,b) = a * b'
Random.seed!(1234)

# Here we go through an example where the theory "works perfectly"

# Define a linear forward map
# input Space 
Mo = 2
# output space
M = 100
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
# timestep size (h = 1/N implies T=1 at the final time)
h = 1/N 
# number of ensemble members
J = 10*Mo
# Likelihood perturbation for algorithm, (constructed from before)
ξ = MvNormal(1/h * Γ)

# Construct empirical prior
u = [rand(prior) for i in 1:J]
u₀ = copy(u)

timeseries = []
push!(timeseries, copy(u))

# Viewed as a minimization problem, we are minimizing
# dot(y - H * x, Γ⁻¹ * (y - H * x)) + dot(x, Σ⁻¹ x)
# whose solution is 
# (H' Γ⁻¹ H + Σ⁻¹)⁻¹ H' Γ⁻¹ y
# Alternatively we can view the posterior as  
# posterior ∝ exp(-dot(y - H * x, Γ⁻¹ * (y - H * x)) - x' Σ⁻¹ x)
# which implies μ = (H' Γ⁻¹ H + Σ⁻¹)⁻¹ H' Γ⁻¹ y
# and           σ = H' Γ⁻¹ H + Σ⁻¹

# EKI Algorithm
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
    push!(timeseries, copy(u))
end

# This 
approx = mean(u)
if length(approx) < 4
    println("mean appro = ", approx)
end
# should converge to this
HΓH = Symmetric((H'* (cholesky(Γ) \ H) + cholesky(Σ) \ I))
cHΓH = cholesky(HΓH) # LU fails
rhs = (H'*(cholesky(Γ)\y̅[:]))
exact = cHΓH\rhs
if length(exact) < 4
    println("mean exact = ", exact) 
end
println("mean relative error = ", norm(approx - exact) / norm(exact))
println("loss relative error = ", norm(H*approx - H*exact) / norm(H*exact))

exact_cov = cholesky(cHΓH) \ I
approx_cov = cov(u)
if length(exact) < 4
    @show approx_cov
    @show exact_cov
end
println("cov relative error = ", norm(approx_cov - exact_cov) / norm(exact_cov) ) 

##
using GLMakie

#  fig = Figure()
#  ax = Axis(fig)
fig, ax, sc = scatter([(timeseries[1][i][1], timeseries[1][i][2]) for i in eachindex(u)], color = :red)
scatter!(ax, [(timeseries[end][i][1], timeseries[end][i][2]) for i in eachindex(u)], color = :blue)
scatter!(ax, [(exact[1], exact[2])], marker = '⋆', color = :yellow, markersize = 30)
display(fig)

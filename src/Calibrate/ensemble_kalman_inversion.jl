#=
struct EnsembleKalmanInversion{S,T}
    iterations::S
    ensemble::T
end
=#

# optimize(f, ensemble, ::EnsembleKalmanInversion)
using Random, LinearAlgebra, Distributions
Random.seed!(1234)
# input Space 
Mo = 3
# output space
M = 3

A = randn(M,Mo) 
N = 1 # number of timesteps
h = 1/N # timestep size 1 / N to converge to posterir
H = A
x = randn(Mo) .- 10 # solution
y̅ = H * x

# J = 10 * Mo # 10x the input space 
# J * N is the number of times the forward map is evaluated
J = 100

⊗(a,b) = a * b'

forward_map(x) = H * x
MM = randn(M,M)
Γ = MM'*MM + I
ξ = MvNormal(1/h * Γ)

# Initial Condition 
MoMo = randn(Mo,Mo)
Σ =  0*MoMo' * MoMo + I 
prior = MvNormal(Σ)
u = [rand(prior) for i in 1:J]
u₀ = copy(u)
# u, forward_map, N, J, h, ξ(h, Γ), y̅, Γ
# Algorithm (Naive), StructArrays may clean this up a bit
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
HΓH = Symmetric((H'*(Γ\H) + cholesky(Σ) \ I))
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
# println("appro cov = ", approx_cov)
# println("exact cov = ", exact_cov)

#=
fig, ax, sc = scatter([(u[i][1], u[i][2]) for i in eachindex(u)], color = :red)

scatter!(ax, [(u₀[i][1], u₀[i][2]) for i in eachindex(u)], color = :blue)
scatter!(ax, [(exact[1], exact[2])], marker = '⋆', color = :yellow, markersize = 30)
display(fig)
=#
##
# calibrate(forward_map, [], )


##
function eki_function!(u, N, forward_map, J, y̅, ξ, Γ)
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
end




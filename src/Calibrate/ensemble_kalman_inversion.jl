using Distributions, Random, LinearAlgebra
# This is a bad design change it
@Base.kwdef struct EKI{A, B, C, D, E}
    Γ::A 
    Σ::B 
    N::C 
    Δt::D
    J::E
end
# include random_seed as a part of eki
# Σ is not used. It is the regularization covariance

# EKI() = EKI([nothing for i in fieldnames(EKI)]...)

#=
using GLMakie
fig, ax, sc = scatter([(u[i][1], u[i][2]) for i in eachindex(u)], color = :red)

scatter!(ax, [(u₀[i][1], u₀[i][2]) for i in eachindex(u)], color = :blue)
scatter!(ax, [(exact[1], exact[2])], marker = '⋆', color = :yellow, markersize = 30)
display(fig)
=#

##
⊗(a,b) = a * b'

"""
eki_step!(u, forward_map, y̅, J, ξ, Γ, Δt)

# Description 
Take a single step of the Ensemble Kalman Inversion (EKI) algorithm
# Langevin formulation
du = Cᵘᵖ Γ⁻¹(y- G(u)) dt + Cᵘᵖ Γ^(-1/2) dW
# Fokker-Planck formulation
∂ₜρ = - ∇ᵤ⋅([y- G(u)]ᵀΓ⁻¹Cov(u, G(u)) + 1/2 Tr(Cov(u, G)Γ⁻¹Cov(u, G) Hᵤ(ρ) )
where the covariances are calculated using the instantaneous density
and Hᵤ is the Hessian of ρ. Note the location of the diffusivity 
# Heuristically it is not ∇⋅(κ∇ρ) but rather Trace(κ∇⊗∇ρ)

# Arguments 
`u`: Vector{S} where S. OVERWRITTEN. Empirical prior distribution 
`forward_map`: Function. Operates on objects of type S
`data`: Vector or Array
`J`: int, number of ensemble members 
`ξ`: multivariate distribution,
`Γ`: matrix corresponding to regularization 
`Δt`: timestep size

# Return 
nothing
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
    return nothing
end

function eki_loop!(u, forward_map, y̅, J, Γ, Δt, N; random_seed = 1234)
    Random.seed!(random_seed)
    ξ = MvNormal(1/Δt * Γ)
    for n in 1:N
        eki_step!(u, forward_map, y̅, J, ξ, Γ, Δt)
    end
    for callback in callbacks 
        callback(u)
    end
end

function calibrate!(x, G, y, method::EKI)
    (; J, Γ, Δt, N) = method
    eki_loop!(G, x, y, J, Γ, Δt, N; random_seed = 1234)
end

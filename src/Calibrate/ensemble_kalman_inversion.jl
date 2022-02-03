using Distributions, Random, LinearAlgebra
# This is a bad design change it
Base.@kwdef struct EKI{A,B,C,D,E}
    Γ::A
    Σ::B
    N::C
    Δt::D
    J::E
end

⊗(a, b) = a * b'

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
    Cᵘᵖ *= 1 / J
    Cᵖᵖ *= 1 / J

    y = [data + rand(ξ) for j in 1:J]
    residual = y - G

    factored_Cᵖᵖ = cholesky(Symmetric(Cᵖᵖ + 1 / Δt * Γ))
    for j in 1:J
        u[j] += Cᵘᵖ * (factored_Cᵖᵖ \ residual[j])
    end
    return nothing
end

function eki_loop!(u, forward_map, y̅, J, Γ, Δt, N)
    ξ = MvNormal(1 / Δt * Γ)
    for n in 1:N
        eki_step!(u, forward_map, y̅, J, ξ, Γ, Δt)
    end
end

function calibrate!(x, G, y, method::EKI)
    (; J, Γ, Δt, N) = method
    eki_loop!(G, x, y, J, Γ, Δt, N)
end

function two_player_eki_step!(u, forward_map, data, J, ξ, Γ, Δt)
    u̅ = mean(u)
    G = forward_map.(u) # error handling needs to go here
    G̅ = mean(G)

    Cᵘᵘ = (u[1] - u̅) ⊗ (u[1] - u̅)
    Cᵘᵖ = (u[1] - u̅) ⊗ (G[1] - G̅)
    Cᵖᵖ = (G[1] - G̅) ⊗ (G[1] - G̅)
    for j in 2:J
        Cᵘᵘ += (u[j] - u̅) ⊗ (u[j] - u̅)
        Cᵘᵖ += (u[j] - u̅) ⊗ (G[j] - G̅)
        Cᵖᵖ += (G[j] - G̅) ⊗ (G[j] - G̅)
    end
    Cᵘᵘ *= 1 / J
    Cᵘᵖ *= 1 / J
    Cᵖᵖ *= 1 / J

    y = [data + rand(ξ) for j in 1:J]
    residual = y - G

    factored_Cᵖᵖ = cholesky(Symmetric(Cᵖᵖ + 1 / Δt * Γ))
    for j in 1:J
        u[j] += Cᵘᵖ * (factored_Cᵖᵖ \ residual[j])
    end

    for j in div(J, 2):J
        u[j] += logpdf(MvNormal(u̅, Cᵘᵘ), u[j])
    end

    return nothing
end

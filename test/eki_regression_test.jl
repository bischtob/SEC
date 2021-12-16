using SEC, Random, Distributions, LinearAlgebra
import SEC.Calibrate: EKI, calibrate!
import Distributions: MvNormal
import LinearAlgebra: I

Random.seed!(1234)

Mo = 2
M = 2
H = randn(M,Mo) 
forward_map(x) = H * x

x = randn(Mo) .- 10 
y̅ = forward_map(x)

MoMo = randn(Mo,Mo)
Σ = MoMo' * MoMo + I 
prior = MvNormal(Σ)

MM = randn(M,M)
Γ = MM'*MM + I

N = 1 
h = 1/N 
J = 1000
ξ = MvNormal(1/h * Γ)

u = [rand(prior) for i in 1:J]
u₀ = copy(u)

Δt = h
u = copy(u₀)

beki = EKI(Γ, Σ, N, Δt, J)
calibrate!(forward_map, u, y̅, beki)
[parameters for i in 1:Number_of_members]
approx = mean(u)
if length(approx) < 4
    println("mean appro = ", approx)
end
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


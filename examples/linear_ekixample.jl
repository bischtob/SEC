using Random, LinearAlgebra, Distributions
import Distributions: MvNormal
import LinearAlgebra: cholesky, Symmetric

⊗(a, b) = a * b'
Random.seed!(1234)

# Here we go through an example where the theory "works perfectly"

# Define a linear forward map
# input Space 
Mo = 2
# output space
M = 2
H = randn(M, Mo)
forward_map(x) = H * x

# Create artificial solution
x = randn(Mo) .- 30
y̅ = forward_map(x)

# Define an initial ensemble distribution
MoMo = randn(Mo, Mo)
Σ = MoMo' * MoMo + I
prior = MvNormal(Σ)
# this choice implies prior ∝ exp( - x' Σ⁻¹ x)

# Implicitly define the likelihood function via the covariance 
MM = randn(M, M)
Γ = (MM' * MM + I)
# implies likelihood  exp(- T * dot(y - H * x, Γ⁻¹ * (y - H * x)) )
# where T is the final time of the simulation
# Observe that
# likelihood = exp(- dot(y - H * x, Γ⁻¹ * (y - H * x)) ) for T = 1

# number of steps
N = 100
# timestep size (h = 1/N implies T=1 at the final time)
h = 1 / N
# number of ensemble members
J = 4000 * Mo
# Likelihood perturbation for algorithm, (constructed from before)
ξ = MvNormal(1 / h * Γ)

# Construct Posterior
HΓHΣ = Symmetric((H' * (cholesky(Γ) \ H) + cholesky(Σ) \ I))
cHΓHΣ = cholesky(HΓHΣ) # LU fails
rhs = (H' * (cholesky(Γ) \ y̅[:]))
posterior_μ = cHΓHΣ \ rhs
posterior_Σ = cHΓHΣ \ I
posterior = MvNormal(posterior_μ, posterior_Σ)

# Construct empirical prior
u = [rand(prior) for i = 1:J]
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
for i = 1:N
    u̅ = mean(u)
    G = forward_map.(u) # error handling needs to go here
    G̅ = mean(G)

    # define covariances
    Cᵘᵖ = (u[1] - u̅) ⊗ (G[1] - G̅)
    Cᵖᵖ = (G[1] - G̅) ⊗ (G[1] - G̅)
    for j = 2:J
        Cᵘᵖ += (u[j] - u̅) ⊗ (G[j] - G̅)
        Cᵖᵖ += (G[j] - G̅) ⊗ (G[j] - G̅)
    end
    Cᵘᵖ *= 1 / (J - 1)
    Cᵖᵖ *= 1 / (J - 1)

    # ensemblize the data
    y = [y̅ + rand(ξ) for i = 1:J]
    r = y - G

    # update
    Cᵖᵖ_factorized = cholesky(Symmetric(Cᵖᵖ + 1 / h * Γ))
    for j = 1:J
        u[j] += Cᵘᵖ * (Cᵖᵖ_factorized \ r[j])
    end
    push!(timeseries, copy(u))
end

# This 
approx = mean(u)
if length(approx) < 4
    println("mean appro = ", approx)
end
# should converge to this
exact = posterior.μ
if length(exact) < 4
    println("mean exact = ", exact)
end
println("mean relative error = ", norm(approx - exact) / norm(exact))
println("loss relative error = ", norm(H * approx - H * exact) / norm(H * exact))

exact_cov = posterior.Σ
approx_cov = cov(u)
if length(exact) < 4
    @show approx_cov
    @show exact_cov
end
println("cov relative error = ", norm(approx_cov - exact_cov) / norm(exact_cov))

##
using GLMakie, Printf

fig = Figure()
ax = Axis(fig[2, 1])
sc_init = scatter!(ax, [(timeseries[1][i][1], timeseries[1][i][2]) for i in eachindex(u)], color = :red)
sc_final = scatter!(ax, [(timeseries[end][i][1], timeseries[end][i][2]) for i in eachindex(u)], color = :blue)

time_slider = Slider(fig, range = 1:length(timeseries), startvalue = 1)
ti = time_slider.value

ensemble = @lift [(timeseries[$ti][i][1], timeseries[$ti][i][2]) for i in eachindex(u)]
sc_transition = scatter!(ax, ensemble, color = :purple)

scatter!(ax, [(exact[1], exact[2])], marker = '⋆', color = :yellow, markersize = 30)

ax.xlabel = "c¹"
ax.ylabel = "c²"

time_string = @lift("Time = " * @sprintf("%0.2f", ($ti - 1) / (length(timeseries) - 1)))
fig[3, 1] = vgrid!(
    Label(fig, time_string, width = nothing),
    time_slider,
)

c¹ = @lift([timeseries[$ti][i][1] for i in eachindex(timeseries[end])])
c² = @lift([timeseries[$ti][i][2] for i in eachindex(timeseries[end])])

ax.xlabelsize = 25
ax.ylabelsize = 25

ax_above = Axis(fig[1, 1])
ax_side = Axis(fig[2, 2])

ax_above.ylabel = "probability density"
ax_side.xlabel = "probability density"

limits!(ax, -50, 10, -8, 4)
xlims!(ax_above, ax.limits[][1]...)
ylims!(ax_above, 0, 0.4)
ylims!(ax_side, ax.limits[][2]...)

hideydecorations!(ax_side, ticks = false, grid = false)
hidexdecorations!(ax_above, ticks = false, grid = false)

# colsize!(fig.layout, 1, Relative(2 / 3))
# rowsize!(fig.layout, 1, Relative(1 / 3))
colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)

d1 = density!(ax_above, c¹, color = (:purple, 0.15), label = "density c¹", strokewidth = 1)
d1lims = ax_above.limits[][1][1]:ax_above.limits[][1][2]
tmpx = collect(range(d1lims[1], d1lims[end], length = 1000))
σ = prior.Σ[1, 1]
μ = prior.μ[1]
tmpy = @. 1 / sqrt(2 * π * σ) * exp(-0.5 * (tmpx - μ)^2 / σ)
ln1 = lines!(ax_above, tmpx, tmpy, linewidth = 4, color = :red)
σ = posterior.Σ[1, 1]
μ = posterior.μ[1]
tmpy = @. 1 / sqrt(2 * π * σ) * exp(-0.5 * (tmpx - μ)^2 / σ)
lines!(ax_above, tmpx, tmpy, linewidth = 4, color = :blue)

d2 = density!(ax_side, c², color = (:purple, 0.15), label = "density c²", strokewidth = 1, direction = :y)
dlims = ax_side.limits[][2][1]:ax_side.limits[][2][2]
tmpx = collect(range(dlims[1], dlims[end], length = 1000))
σ = prior.Σ[2, 2]
μ = prior.μ[2]
tmpy = @. 1 / sqrt(2 * π * σ) * exp(-0.5 * (tmpx - μ)^2 / σ)
lines!(ax_side, tmpy, tmpx, linewidth = 4, color = :red)
σ = posterior.Σ[2, 2]
μ = posterior.μ[2]
tmpy = @. 1 / sqrt(2 * π * σ) * exp(-0.5 * (tmpx - μ)^2 / σ)
ln2 = lines!(ax_side, tmpy, tmpx, linewidth = 4, color = :blue)


Legend(fig[1, 2],
    [sc_init, sc_final, sc_transition, ln1, ln2],
    ["prior ensemble", "posterior ensemble", "transition ensemble", "exact marginal prior", "exact marginal posterior"])

update!(fig.scene)
display(fig)

seconds = 5
fps = 30
frames = round(Int, fps * seconds)
frames = 101
fps = 30
record(fig, pwd() * "/example.mp4"; framerate = fps) do io
    for i = 1:frames
        ti[] = i
        sleep(1 / fps)
        recordframe!(io)
    end
end

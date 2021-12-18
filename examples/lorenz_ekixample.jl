# Here we go through an example using the Lorenz equations
using Random, Distributions, LinearAlgebra
const plotit = false
Random.seed!(1234)

# Initialize arrays for timestepping
ṡ = [[0.0] for i = 1:3]
s = [[randn() + 10] for i = 1:3]
s̃ = copy(s)

function lorenz!(ṡ, s; ρ = 28, β = 8 / 3, σ = 10)
    ẋ, ẏ, ż = ṡ
    x, y, z = s
    @. ẋ = σ * (-x + y)
    @. ẏ = -y + (ρ - z) * x
    @. ż = -β * z + x * y
    return nothing
end

function closure_lorenz!(ρ, β, σ)
    function rhs!(ṡ, s)
        lorenz!(ṡ, s, ρ = ρ, β = β, σ = σ)
    end
end

function step!(s, ṡ, s̃, rhs!, Δt)
    rhs!(ṡ, s)
    @. s̃ = s + Δt * ṡ
    @. s = s + Δt * 0.5 * ṡ
    rhs!(ṡ, s̃)
    @. s = s + Δt * 0.5 * ṡ
    return nothing
end

# timestep and record solution 
timeseries = []
push!(timeseries, [(s...)...])
Δt = 0.005
N = 2^19 - 1

for i = 1:N
    step!(s, ṡ, s̃, lorenz!, Δt)
    push!(timeseries, [(s...)...])
end


x = [timeseries[i][1] for i in eachindex(timeseries)]
y = [timeseries[i][2] for i in eachindex(timeseries)]
z = [timeseries[i][3] for i in eachindex(timeseries)]
#=
if plotit
using GLMakie, Printf, LaTeXStrings
fig = Figure(resolution = (1600, 1300))

averaging_slider = Slider(fig, range = 1:16, startvalue = 1)
i = averaging_slider.value
x̅ₜ = @lift(mean(reshape(x, (2^$i, 2^(19 - $i))), dims = 1)[:])
time_string = @lift("Averaging Interval T = " * @sprintf("%0.3f", $i * Δt))

ax = Axis(fig[1, 1],
    xlabel = L" \frac{1}{\tau+T}\int_{\tau}^T x(t) dt ", xlabelsize = 40,
    ylabel = "pdf", ylabelsize = 40,
    xticklabelsize = 30, yticklabelsize = 30)

@lift ylims!(ax, (0, 2 / sqrt(2 * π * var($x̅ₜ))))
xlims!(ax, (-23, 23))


d1 = density!(ax, x̅ₜ, color = (:purple, 0.15), strokewidth = 1)


fig[2, 1] = vgrid!(
    Label(fig, time_string, width = nothing, textsize = 30),
    averaging_slider,
)

display(fig)

frames = 16
fps = 5
record(fig, pwd() * "/lorenz_ekixample.mp4"; framerate = fps) do io
    for ii = 1:frames
        i[] = ii
        sleep(1 / fps)
        recordframe!(io)
    end
end
end
=#
## 
function closure_forward_map(i)
    function forward_map(C)
        ρ, σ, β = C
        rhs! = closure_lorenz!(ρ, β, σ)
        Δt = 0.005
        N = 2^16 - 1

        Random.seed!(1234)
        # Initialize arrays for timestepping
        ṡ = [[0.0] for i = 1:3]
        s = [[randn() + 10] for i = 1:3]
        s̃ = copy(s)
        timeseries = []
        push!(timeseries, [(s...)...])

        for i = 1:N
            step!(s, ṡ, s̃, rhs!, Δt)
            push!(timeseries, [(s...)...])
        end

        x = [timeseries[i][1] for i in eachindex(timeseries)]
        y = [timeseries[i][2] for i in eachindex(timeseries)]
        z = [timeseries[i][3] for i in eachindex(timeseries)]

        # use the Nusselt number as the observation
        Nuₜ = mean(reshape(z .* β, (2^i, 2^(16 - i))), dims = 1)[:]
        return Nuₜ
    end
end

𝒢 = closure_forward_map(12)
C = [28, 10, 8 / 3]

forward_map(C) = mean(𝒢(C))

y = 𝒢(C)
y̅ = mean(y)
Γ = var(y)
J = length(C) * 10
ξ = Normal(0, Γ)
# Guassianize

prior = MvNormal([28, 10, 8 / 3], Diagonal([3.0, 3.0, 1.0]))
prior = MvNormal([20, 5, 4], Diagonal([5.0, 4.0, 1.0]))
# Construct empirical prior
u = [rand(prior) for i = 1:J]
u₀ = copy(u)

timeseries = []
push!(timeseries, copy(u))

N = 4
h = 1 / N
⊗(a, b) = a * b'
for i = 1:N
    println("loop ", i)
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
    if length(Γ) > 1
        Cᵖᵖ_factorized = cholesky(Symmetric(Cᵖᵖ + 1 / h * Γ))
    else
        Cᵖᵖ_factorized = Cᵖᵖ + 1 / h * Γ
    end
    for j = 1:J
        u[j] += Cᵘᵖ * (Cᵖᵖ_factorized \ r[j])
    end
    push!(timeseries, copy(u))
end

residual = C - mean(timeseries[end])

println("The relative error of ρ is ", abs(residual[1]) / C[1])
println("The relative error of σ is ", abs(residual[2]) / C[2])
println("The relative error of β is ", abs(residual[3]) / C[3])

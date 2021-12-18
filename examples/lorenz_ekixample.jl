# Here we go through an example using the Lorenz equations
using Random
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

using GLMakie, Printf, LaTeXStrings
fig = Figure(resolution = (1600, 1300))
ax = Axis(fig[1, 1],
    xlabel = L" \frac{1}{\tau+T}\int_{\tau}^T x(t) dt ", xlabelsize = 40,
    ylabel = "pdf", ylabelsize = 40,
    xticklabelsize = 30, yticklabelsize = 30)

averaging_slider = Slider(fig, range = 1:16, startvalue = 1)
i = averaging_slider.value
x̅ₜ = @lift(mean(reshape(x, (2^$i, 2^(19 - $i))), dims = 1)[:])

xlims!(ax, (-23, 23))

@lift ylims!(ax, (0, 2 / sqrt(2 * π * var($x̅ₜ))))
d1 = density!(ax, x̅ₜ, color = (:purple, 0.15), strokewidth = 1)
# ax.xlabel =
# hist!(ax, x̅ₜ, color = (:purple, 0.15), strokewidth = 1)

time_string = @lift("Averaging Interval T = " * @sprintf("%0.3f", $i * Δt))
fig[2, 1] = vgrid!(
    Label(fig, time_string, width = nothing, textsize = 30),
    averaging_slider,
)

display(fig)

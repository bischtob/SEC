using SEC, Test, GLMakie, LinearAlgebra
import SEC.Emulate: mean, cov, GaussianProcess, condition

μ(x) = 0.0
δ(x,y) = x==y ? 1 : 0
k(x,y) = 10exp(-(x-y)^2/2) + 0.1 * δ(x,y) # noise here
gp = GaussianProcess(mean = μ, covariance = k)
N = 10
X = [i for i in 1:N]
Y = [2*i for i in 1:N]
data = (; X, Y)
cgp = condition(gp, data, noise = 0.0) # noise here
# philosophical difference only
# just associates more noise at the given prediction point
##
fig = Figure(resolution = (700, 450))
ax = Axis(fig, xlabel = "x", ylabel = "y")
Xn = collect(range(0, N+1, 100))
Xn = setdiff(Xn, X)
μX = mean(cgp, Xn)
σXX =  [cgp.covariance(x,x) for x in Xn]
scatter!(ax, X, Y, color = :red)
lines!(ax, Xn, μX, color = :blue, linewidth =3)
# filled curve 1
band!(ax, Xn, μX - 3*σXX, μX + 3*σXX, color = (:blue, 0.2))
xlims!(ax, (0,11))
ylims!(ax, (-1,21))
ax.xlabelsize = 25
ax.xticklabelsize = 25
ax.ylabelsize = 25 
ax.yticklabelsize = 25
update!(fig.scene)
fig[1,1] = ax
fig

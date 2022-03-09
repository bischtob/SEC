

ρ, σ, β = (28, 10, 8 / 3)
rhs! = closure_lorenz!(ρ, β, σ)
Δt = 0.005
N = 2^22 - 1

Random.seed!(1234)
# Initialize arrays for timestepping
ṡ = [[0.0] for i = 1:3]
s = [[randn() + 10] for i = 1:3]
s̃ = copy(s)
timeseries = []
push!(timeseries, [(s...)...])


for i = 1:N
    step!(s, ṡ, s̃, rhs!, Δt)
    # s .+= 0.2 * sqrt(Δt) * [[randn()] for i in 1:3]
    push!(timeseries, [(s...)...])

end

##
holding_time = []
for i in eachindex(timeseries)
    state = timeseries[i]
    if (sign(state[1]) == sign(state[2]))
        push!(holding_time, Δt)
    elseif (sign(state[1]) != sign(state[2])) & (state[3] > 20)
        push!(holding_time, -Δt)
    end
end

##
left_lobe = [Δt]
right_lobe = [Δt]
jleft = 1
jright = 1
for i in 2:length(holding_time)
    if i % 10000 == 0
        println("jleft is ", jleft)
        println("jright is ", jright)
    end
    ht_before = holding_time[i-1]
    ht_current = holding_time[i]
    if (ht_before > 0) & (ht_current > 0)
        right_lobe[jright] += Δt
    elseif (ht_before < 0) & (ht_current < 0)
        left_lobe[jleft] += Δt
    elseif (ht_before > 0) & (ht_current < 0)
        jleft += 1
        push!(left_lobe, Δt)
    elseif (ht_before < 0) & (ht_current > 0)
        jright += 1
        push!(right_lobe, Δt)
    else
        println("huh?")
    end

end
left_lobe = left_lobe[2:end]
right_lobe = right_lobe[2:end]
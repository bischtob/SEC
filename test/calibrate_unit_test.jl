using SEC, Test
import SEC.Calibrate.optimize
import SEC.Calibrate.Optim: NelderMead, LBFGS, GradientDescent

@testset "Optim.jl" begin
    solution = [1.0, 2.0]
    f(x⃗) = (solution[1] - x⃗[1])^2 + (solution[2] - x⃗[2])^2
    function ∇f!(∇f, x⃗)
        ∇f[1] = -2.0 * (solution[1] - x⃗[1])
        ∇f[2] = -2.0 * (solution[2] - x⃗[2])
    end

    optim_output = optimize(f, [0.0, 0.0], NelderMead())
    (; minimizer) = optim_output
    tolerance = 2e-4
    @test all(abs.(minimizer - solution) .≤ tolerance)

    optim_output = optimize(f, [0.0, 0.0], LBFGS())
    (; minimizer) = optim_output
    tolerance = 2e-11
    @test all(abs.(minimizer - solution) .≤ tolerance)

    optim_output = optimize(f, ∇f!, [0.0, 0.0], GradientDescent())
    (; minimizer) = optim_output
    tolerance = 2e-11
    @test all(abs.(minimizer - solution) .≤ tolerance)
end
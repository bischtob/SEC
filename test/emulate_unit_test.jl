using SEC, Test 
import SEC.Emulate: mean, cov, GaussianProcess

@testset "GaussianProcess unit test" begin
    μ(x) = 0.0
    k(x,y) = exp(-(x-y)^2/2)
    gp = GaussianProcess(mean = μ, covariance = k)
    @test μ === mean(gp)
    @test k === cov(gp)
end

@testset "1D GPR mean and cov data test" begin
    μ(x) = 0.0
    k(x,y) = exp(-(x-y)^2/2)
    gp = GaussianProcess(mean = μ, covariance = k)
    N = 10
    X = [i for i in 1:N]
    μX = mean(gp, X)
    σX = cov(gp, X)
    @test all(μX .== zeros(Float64, N))
    @test σX[1,1] ≈ 1.0
    @test σX[1, N] ≈ k(1, N)
end

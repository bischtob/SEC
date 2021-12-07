using SEC
using Test

@testset "Hello World " begin
    @test isnothing(SEC.Calibrate.greet())
    @test isnothing(SEC.Emulate.greet())
    @test isnothing(SEC.Sample.greet())
end
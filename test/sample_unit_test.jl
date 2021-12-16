using SEC, Test, Statistics, Distributions, Random
import SEC.Sample: markov_link, markov_chain

@testset "markov link correctness" begin
    nll(X) = 10
    proposal(X) = 2
    current_X = 1
    current_nll = nll(current_X)
    (; new_X, new_nll, proposal_X, proposal_nll) = markov_link(
        nll = nll,
        proposal = proposal,
        current_nll = current_nll,
        current_X = current_X
    )
    @test new_X == 2
    @test new_nll == 10
    @test proposal_X == 2
    @test proposal_nll == 10
end

@testset "markov chain reproducibility" begin
    nll(X) = X^2 / 2
    proposal(X) = X + randn()
    seed_X = 0.0
    chain_length = 10000
    (; chain_X, chain_nll) = markov_chain(nll, proposal, seed_X, chain_length, random_seed = 1234)
    original_chain_X = copy(chain_X)
    original_chain_nll = copy(chain_nll)
    (; chain_X, chain_nll) = markov_chain(nll, proposal, seed_X, chain_length, random_seed = 1234)
    @test all((original_chain_X - chain_X) .== 0.0)
    @test all((original_chain_nll - chain_nll) .== 0.0)
    # hand unrolling loop
    Random.seed!(1234)
    current_X = 0.0
    current_nll = nll(current_X)
    proposal_X = proposal(seed_X)
    proposal_nll = nll(proposal_X)
    Δ = current_nll - proposal_nll
    accept_bool = log(rand(Uniform(0, 1))) < Δ
    current_X = accept_bool ? proposal_X : current_X
    @test chain_X[2] == current_X
end

@testset "markov chain convergence" begin
    nll(X) = X^2 / 2
    proposal(X) = X + randn()
    seed_X = 0.15
    chain_length = 10000
    (; chain_X, chain_nll) = markov_chain(nll, proposal, seed_X, chain_length, random_seed = 1234)
    @test abs(mean(chain_X)) < 1 / sqrt(chain_length)
    @test abs(std(chain_X) - 1.0) < +1 / sqrt(chain_length)
end




  
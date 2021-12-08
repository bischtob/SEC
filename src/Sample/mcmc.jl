"""
acceptance criteria for Metropolis-Hastings
# Definition 
accept(Δ) = log(rand(Uniform(0, 1))) < Δ
# Arguments
- `Δ`: (scalar): change in negative log-likelihood 
# Output
- true or false (bool)
# Notes
Always go downhill but sometimes go uphill
"""
accept(Δ) = log(rand(Uniform(0, 1))) < Δ

"""
markov_link(nll, proposal, current_X, current_nll)
# Description
- Takes a single step in the random walk markov chain monte carlo algorithm and outputs proposal parameters, 
  new parameters, and the evaluate of the loss function
# Arguments
- `nll`: The negative log-likelihood function. In the absence of priors this becomes a loss function
- `proposal`: (function), determines the proposal step
- `current_X`: (array), current parameter
- `current_nll`: (scalar), proposal_nll = nll(X). The value of negative log-likelihood of the current parameter
# Return
- `new_X`: The value of the accepted X
- `new_nll`: value of nll(new_X)
- `proposal_X`: The X from the "proposal step". Was either rejected or accepted.
- `proposal_nll`: value of nll(proposal_X)
"""
function markov_link(; nll, proposal, current_X, current_nll)
    proposal_X = proposal(current_X)
    proposal_nll = nll(proposal_X)
    Δ = (current_nll - proposal_nll)

    if accept(Δ)
        new_X, new_nll = proposal_X, proposal_nll
    else
        new_X, new_nll = current_X, current_nll
    end

    return (; new_X, new_nll, proposal_X, proposal_nll)
end

"""
markov_chain(nll, proposal, seed_X, chain_length; random_seed = 1234)
# Description
- A random walk that computes the posterior distribution
# Arguments
- `nll`: The negative log-likelihood function. In the absence of priors this becomes a loss function
- `proposal`: (function), proposal function for MCMC
- `seed_X`: (Array), initial parameter values
- `chain_length`: (Int) number of markov chain monte carlo steps
- `perturb`: a function that performs a perturbation of X
# Keyword Arguments
- `random_seed`: determines the seed of the random number generator
# Return
- `chain_X`: The matrix of accepted parameters in the random walk
- `chain_nll`: The array of errors associated with each step in param chain
"""
function markov_chain(nll, proposal, seed_X, chain_length::Int; random_seed = 1234)
    Random.seed!(random_seed)

    current_X = seed_X
    current_nll = nll(seed_X)
    chain_X = typeof(current_X)[]
    chain_nll = typeof(current_nll)[]
    push!(chain_X, current_X)
    push!(chain_nll, current_nll)

    for i = 1:chain_length-1
        (; new_X, new_nll) = markov_link(; nll, proposal, current_X, current_nll)
        current_X, current_nll = new_X, new_nll # mcmc update
        push!(chain_X, new_X)
        push!(chain_nll, new_nll)
    end

    return (; chain_X, chain_nll)
end
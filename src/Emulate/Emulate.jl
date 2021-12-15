module Emulate
using Distributions, Random, LinearAlgebra

import Distributions: rand
import Distributions: mean, cov

include("gaussian_processes.jl")

greet() = println("Hello World!")

end # module
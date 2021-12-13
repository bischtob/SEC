module Emulate
using Distributions, Random, LinearAlgebra

import Distributions: rand
import Distributions: mean, cov

include("emulator.jl")

greet() = println("Hello World!")

end # module
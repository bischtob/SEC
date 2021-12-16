module Emulate
using Distributions
using Random
using LinearAlgebra
using Flux
using FFTW
using Zygote

using Tullio
using CUDA
using CUDAKernels
using KernelAbstractions
using ChainRulesCore

import Distributions: rand
import Distributions: mean, cov

include("gaussian_processes.jl")

greet() = println("Hello World!")

end # module
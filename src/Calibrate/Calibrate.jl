module Calibrate
using Optim
import Optim: optimize

export optimize

include("ensemble_kalman_inversion.jl")

greet() = println("Hello World!")

end # module
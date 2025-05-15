module QuanticsTCI

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

import LinearAlgebra: rank
import Base: sum

export quanticscrossinterpolate, evaluate, sum, integral
export cachedata, quanticsfouriermpo

import BitIntegers
import BitIntegers: UInt256, UInt512, UInt1024

BitIntegers.@define_integers 2048 MyInt2048 MyUInt2048

include("tciinterface.jl")
include("fouriertransform.jl")

end

module QuanticsTCI

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

import LinearAlgebra: rank
import Base: sum

export quanticscrossinterpolate, evaluate, sum, integral
export cachedata, quanticsfouriermpo

include("tciinterface.jl")
include("fouriertransform.jl")
include("derivative.jl")

end

module QuanticsTCI

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

import LinearAlgebra: rank
import Base: sum

export quanticscrossinterpolate, evaluate, sum, integral
export cachedata

include("tciinterface.jl")

end

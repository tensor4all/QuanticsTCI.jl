module QuanticsTCI

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI

# To add methods to rank
import LinearAlgebra: rank

import QuanticsGrids as QG

export evaluate
export quanticscrossinterpolate
export cachedata

include("tciinterface.jl")

end

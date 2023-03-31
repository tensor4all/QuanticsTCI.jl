# QuanticsTCI

[![Build Status](https://gitlab.com/marc.ritter/QuanticsTCI.jl/badges/main/pipeline.svg)](https://gitlab.com/marc.ritter/QuanticsTCI.jl/pipelines)
[![Coverage](https://gitlab.com/marc.ritter/QuanticsTCI.jl/badges/main/coverage.svg)](https://gitlab.com/marc.ritter/QuanticsTCI.jl/commits/main)

This module contains utilities for interpolations of functions in the quantics TCI / quantics tensor train (QTT) format.

## Installation
Once the module has been published, the following will install QuanticsTCI.jl:

```julia
julia> using Pkg; Pkg.add("QuanticsTCI.jl")
```

This module depends on:
- [TensorCrossInterpolation.jl](https://gitlab.com/quanticstci/tensorcrossinterpolation.jl)
- [ITensors.jl](https://github.com/ITensor/ITensors.jl)

Due to ITensors, Julia 1.6 or newer is required.

---

Until the module is available via `Pkg`, use the following instructions.

1. Clone the repository to some convenient path
```sh
$ cd /convenient/path
$ git clone git@gitlab.com:quanticstci/quanticstci.jl.git
```
2. In a julia REPL, tell julia where you put the downloaded repository.
```julia
julia> using Pkg; Pkg.dev("convenient/path/quanticstci.jl")
```
3. You should now be able to import the module.
```julia
julia> using QuanticsTCI
```
---

## Usage

The main functionality of this package is in the functions `quantics_to_index` and `index_to_quantics`. These translate between linear and quantics representation. For multivariate functions, you have a choice between the *interleaved* and *fused* representation (see QTCI paper [arXiv:2303.11819](http://arxiv.org/abs/2303.11819)). For the *interleaved* representation with `R` bits in each dimension, use
```julia
sigma = index_to_quantics_interleaved([u1, u2, u3], R)
[u1, u2, u3] = quantics_to_index_interleaved(sigma, 3)
```
For the *fused* representation, the methods are very similar:
```julia
sigma = index_to_quantics_fused([u1, u2, u3], R)
[u1, u2, u3] = quantics_to_index_fused(sigma, 3)
```
Further information can be found in the corresponding docstrings.

For convenience, function wrappers are available. If `f` is a function of a single parameter, a quantics version of `f` can be obtained by
```julia
qf = QuanticsFunction{Float64}(f)
```
Note that the return type of `f` (`Float64` in this case) has to be specified. called like a normal function, e.g. 
```julia
value = f([1, 2, 1, 1, 2, 1])
```
For multivariate functions `f`, use `QuanticsFunctionInterleaved` or `QuanticsFunctionFused`. In addition to `f`, specifify the number of dimensions `ndims`:
```julia
qfinterleaved = QuanticsFunctionInterleaved{Float64}(f, ndims)
qffused = QuanticsFunctionFused{Float64}(f, ndims)
```
All of these are suitable for passing to the `TensorCrossInterpolation.crossinterpolate` function. Note that the resulting QTCI / QTT takes parameters in quantics form, e.g.
```
R = 5
f(u) = 1 / (1 + u' * u)
qfinterleaved = QuanticsFunctionInterleaved{Float64}(f, 4)
qtci, ranks, errors = TCI.crossinterpolate(Float64, qfinterleaved, fill(2, 4 * R))
qtt = TCI.TensorTrain(qtci)
value = f([1, 2, 3, 4])
println("Exact value: $value")
qttapprox = qtt(index_to_quantics_interleaved([1, 2, 3, 4], R))
println("QTT approximated value: $qttapprox")
```
A convenience function that encapsulates the above code block in a single call is being worked on.

## Related libraries
- [TensorCrossInterpolation.jl](https://gitlab.com/quanticstci/tensorcrossinterpolation.jl) to calculate tensor cross interpolations.
- [ITensors.jl](https://github.com/ITensor/ITensors.jl) for MPS / MPO algorithms.

## References
- M. K. Ritter, Y. N. Fernández, M. Wallerberger, J. von Delft, H. Shinaoka, and X. Waintal, *Quantics Tensor Cross Interpolation for High-Resolution, Parsimonious Representations of Multivariate Functions in Physics and Beyond*, [arXiv:2303.11819](http://arxiv.org/abs/2303.11819).

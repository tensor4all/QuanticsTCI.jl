# QuanticsTCI

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensors4fields.gitlab.io/quanticstci.jl/dev/index.html)
[![pipeline status](https://gitlab.com/tensors4fields/quanticstci.jl/badges/main/pipeline.svg)](https://gitlab.com/tensors4fields/quanticstci.jl/-/commits/main)
[![coverage report](https://gitlab.com/tensors4fields/quanticstci.jl/badges/main/coverage.svg)](https://gitlab.com/tensors4fields/quanticstci.jl/-/commits/main)

This module contains utilities for interpolations of functions in the quantics TCI / quantics tensor train (QTT) format.

## Installation

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

Once the module has been published, the following will install QuanticsTCI.jl:

```julia
julia> using Pkg; Pkg.add("QuanticsTCI.jl")
```

This module depends on:
- [TensorCrossInterpolation.jl](https://gitlab.com/quanticstci/tensorcrossinterpolation.jl)
- [ITensors.jl](https://github.com/ITensor/ITensors.jl)

Due to ITensors, Julia 1.6 or newer is required.

## Definition
We first introduce a $B$-base presetantion ($b=2, 3, 4, \cdots$).
To avoid confusing, we will use the 1-based indexing of Julia below.
We represent a positive integer $X~(\ge 1)$ as $X= \sum_{i=1}^R (x_i-1) \times B^{R-i+1} + 1$, where $x_i$ is either 1 or 2 and $R$ is the number of digits.
In this library, the $B$-base representation of $X$ is represented by the vector $[x_1, \cdots, x_R]$.

This library supports two unfolding schemes (interleaved and fused representations) for handling multiple variables.
As an example, we consider three variables $X$, $Y$ and $Z$.
Their $B$-base representations are given by  $[x_1, \cdots, x_R]$, $[y_1, \cdots, y_R]$, $[z_1, \cdots, z_R]$, respectively.
The interleaved representation of these variables reads $[x_1, y_1, z_1, x_2, y_2, z_2, \cdots, x_R, y_R, z_R]$.
The fused representation is given by $[\alpha_1, \alpha_2, \cdots, \alpha_R]$, where $\alpha_i = (x_i-1) + B(y_i-1) + B^2 (z_i-1) + 1$ with $1 \le \alpha_i \le B^3$.
This convention is consistent with the column major used in Julia: At each digit level $i$, $x_i$ runs fastest.
The fused representaion generalizes to any number of variables.


## Usage

The main functionality of this package is in the functions `quantics_to_index` and `index_to_quantics`. These translate between linear and quantics representation. For multivariate functions, you have a choice between the *interleaved* and *fused* representation (see QTCI paper [arXiv:2303.11819](http://arxiv.org/abs/2303.11819)). For the *interleaved* representation with `R` bits in each dimension, use
```julia
sigma = index_to_quantics_interleaved([u1, u2, u3], R)
[u1, u2, u3] = quantics_to_index_interleaved(sigma, n)
```
where `n` is the number of dimensions.

For the *fused* representation, the methods are very similar:
```julia
sigma = index_to_quantics_fused([u1, u2, u3], R)
[u1, u2, u3] = quantics_to_index_fused(sigma, n)
```
Further information can be found in the corresponding docstrings.

For convenience, function wrappers are available. If `f` is a function of a single parameter, a quantics version `qf` can be obtained by
```julia
qf = QuanticsFunction{Float64}(f)
```
Note that the return type of `f` (`Float64` in this case) has to be specified. `qf` can be called like a normal function, and takes a list of quantics indices as its only parameter.
```julia
value = f([1, 2, 1, 1, 2, 1])
```
For multivariate functions `f`, use `QuanticsFunctionInterleaved` or `QuanticsFunctionFused`. In addition to `f`, the number of dimensions `ndims` has to be specified:
```julia
qfinterleaved = QuanticsFunctionInterleaved{Float64}(f, ndims)
qffused = QuanticsFunctionFused{Float64}(f, ndims)
```
All of these objects are suitable for passing to the `TensorCrossInterpolation.crossinterpolate` function. A QTCI of a function `f` can be obtained like this:
```julia
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
Note that the resulting QTCI / QTT takes parameters in quantics form.
A convenience function that encapsulates the above code block in a single call is being worked on.

## Related libraries
- [TensorCrossInterpolation.jl](https://gitlab.com/quanticstci/tensorcrossinterpolation.jl) to calculate tensor cross interpolations.
- [ITensors.jl](https://github.com/ITensor/ITensors.jl) for MPS / MPO algorithms.

## References
- M. K. Ritter, Y. N. Fernández, M. Wallerberger, J. von Delft, H. Shinaoka, and X. Waintal, *Quantics Tensor Cross Interpolation for High-Resolution, Parsimonious Representations of Multivariate Functions in Physics and Beyond*, [arXiv:2303.11819](http://arxiv.org/abs/2303.11819).

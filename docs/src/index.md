# QuanticsTCI.jl user guide

```@meta
CurrentModule = QuanticsTCI
```

This module allows easy translation of functions to quantics representation. It meshes well with the `TensorCrossInterpolation.jl` module, together with which it provides quantics TCI functionality.

# Quickstart
The easiest way to construct a quantics tensor train is the `quanticscrossinterpolate` function. For example, the function ``f(x, y) = (cos(x) - cos(x - 2y)) * abs(x + y)`` can be interpolated as follows.

```@example simple
using QuanticsTCI
f(x, y) = (cos(x) - cos(x - 2y)) * abs(x + y)
xvals = range(-6, 6; length=256)
yvals = range(-12, 12; length=256)
qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals, yvals]; tolerance=1e-8)
```
The output object `qtt` now represents a quantics tensor train. It can then be evaluated a function of indices enumerating the `xvals` and `yvals` arrays.

```@example simple
using Plots
qttvals = qtt.(1:256, collect(1:256)')
contour(xvals, yvals, qttvals, fill=true)
xlabel!("x")
ylabel!("y")
savefig("simple.svg"); nothing # hide
```

![](simple.svg)

The convergence criterion can be controlled using the keywords `tolerance`, `pivottolerance`, and `maxbonddim`.
- `tolerance` is the value of the error estimate at which the optimization algorithm will stop.
- `pivottolerance` is the threshold at which each local optimization will truncate the bond.
- `maxbonddim` sets the maximum bond dimension along the links.

A common default setting is to control convergence using `tolerance`, and to set `pivottolerance` equal or slightly smaller than that. Specifying `maxbonddim` can be useful as a safety. However, if `maxbonddim` is set, one should check the error estimate for convergence afterwards.

In the following example, we specify all 3 parameters, but set `maxbonddim` too small.
```@example simple
qtt, ranks, errors = quanticscrossinterpolate(
    Float64, f, [xvals, yvals];
    tolerance=1e-8,
    pivottolerance=1e-8,
    maxbonddim=8)
print(last(errors))
qttvals = qtt.(1:256, collect(1:256)')
contour(xvals, yvals, qttvals, fill=true)
xlabel!("x")
ylabel!("y")
savefig("simpletrunc.svg"); nothing # hide
```
![](simpletrunc.svg)

The plot shows obvious noise due to the insufficient maximum bond dimension. Accordingly, the error estimate of ``0.08`` shows that convergence has not been reached, and an increase of the maximum bond dimension is necessary.

# Further reading

- See the API Reference for all variants of calling [`quanticscrossinterpolate`](@ref).
- If you are having trouble with convergence / efficiency of the TCI, you might have to tweak some of its options. All keyword arguments are forwarded to `TensorCrossInterpolation.crossinterpolate2()` internally. See its [documentation](https://tensor4all.github.io/TensorCrossInterpolation.jl/dev/documentation/#TensorCrossInterpolation.crossinterpolate2-Union{Tuple{N},%20Tuple{ValueType},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}}},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}},%20Vector{Vector{Int64}}}}%20where%20{ValueType,%20N}) for further information.
- If you intend to work directly with the quantics representation, [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl) is useful for conversion between quantics and direct representations. More advanced use cases can be implemented directly using this library.

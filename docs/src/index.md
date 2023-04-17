# QuanticsTCI.jl user guide

```@meta
CurrentModule = QuanticsTCI
```

This module allows easy translation of functions to quantics representation. It meshes well with the `TensorCrossInterpolation.jl` module, together with which it provides quantics TCI functionality.

# Quantics representation

## One dimension  / variable
Functions that translate between "normal" and quantics representation, i.e. between ``i`` and ``u_k`` in
```math
i = \frac{u_1}{1} + \frac{u_2}{2} + \ldots + \frac{u_n}{2^{n-1}}
```
This is effectively just a representation in ``n`` bits, where each bit is a tensor leg of the tensor to be represented.

```@docs
QuanticsTCI.index_to_quantics
QuanticsTCI.quantics_to_index
```

## Multiple dimensions / variables
There are two ways to represent ``D``-dimensional functions in quantics:
1. **Fused representation**: The ``D`` tensor legs corresponding to the same length scale, with local dimension ``d`` (usually ``d=2``) are merged to one leg with dimension ``d^D``.
2. **Interleaved representation**: Tensor legs are interleaved, such that the ``D`` tensor legs corresponding to each length scale are neighbours.

### Translation

To translate between multiple indices and their quantics representation in *fused* representation, use the following functions:
```@docs
QuanticsTCI.index_to_quantics_fused
QuanticsTCI.quantics_to_index_fused
```


To translate between multiple indices and their quantics representation in *interleaved* representation, use the following functions:
```@docs
QuanticsTCI.index_to_quantics_interleaved
QuanticsTCI.quantics_to_index_interleaved
```

### Fusing legs
To fuse legs by hand:
```@docs
QuanticsTCI.fuse_dimensions
QuanticsTCI.split_dimensions
```

### Interleaving legs
To interleave legs:
```@docs
QuanticsTCI.interleave_dimensions
QuanticsTCI.deinterleave_dimensions
```
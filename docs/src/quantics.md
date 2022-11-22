# Quantics representation

## Translators
Functions that translate between "normal" and quantics representation, i.e. between ``i`` and ``u_k`` in
```math
i = \frac{u_1}{1} + \frac{u_2}{2} + \ldots + \frac{u_n}{2^{n-1}}
```
This is effectively just a representation in ``n`` bits, where each bit is a tensor leg of the tensor to be represented.

### Index to Quantics
```@docs
QuanticsTCI.binary_representation
QuanticsTCI.index_to_quantics
```

### Quantics to Index
```@docs
QuanticsTCI.quantics_to_index
```

## Multiple dimensions
There are two ways to represent ``D``-dimensional functions in quantics:
1. Merge dimensions: The ``D`` tensor legs corresponding to the same length scale, with local dimension ``d`` (usually ``d=2``) are merged to one leg with dimension ``d^D``.
2. Interleave dimensions: Tensor legs are interleaved, such that the ``D`` tensor legs corresponding to each length scale are neighbours.

### Merging legs
```@docs
QuanticsTCI.merge_dimensions
QuanticsTCI.split_dimensions
```

### Interleaving legs
```@docs
QuanticsTCI.interleave_dimensions
QuanticsTCI.deinterleave_dimensions
```
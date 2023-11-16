
# Grid for d-dimensional space with base of B
abstract type Grid{d,B} end

"""
Convert a grid index to the corresponding coordinate in the original coordinate system
"""
function grididx_to_origcoord(g::Grid{d,B}, index::NTuple{d,Int}) where {d,B}
    all(1 .<= index .<= B^g.R) || error("1 <= {index} <= B^g.R")
    return (index .- 1) .* grid_step(g) .+ grid_min(g)
end

"""
Convert an grid index to quantices indices
"""
function grididx_to_quantics(g::Grid{d,B}, grididx::NTuple{d,Int}) where {d,B}
    return index_to_fused_quantics(grididx, g.R)
end

"""
Convert fused quantics bitlist to the original coordinate system
"""
function quantics_to_origcoord_fused(g::Grid{d,B}, bitlist) where {d,B}
    idx = quantics_to_index_fused(bitlist; base=Val(B), dims=Val(d))
    return grididx_to_origcoord(g, idx)
end

function quantics_function_fused(::Type{T}, g::Grid{d,B}, f::Function)::Function where {T,d,B}
    function _f(bitlist)::T
        return f(quantics_to_origcoord_fused(g, bitlist)...)
    end
    return _f
end


@doc raw"""
The InherentDiscreteGrid struct represents a grid for inherently discrete data.
The grid contains values at specific, 
equally spaced points, but these values do not represent discretized versions 
of continuous data. Instead, they represent individual data points that are 
inherently discrete.
The linear size of the mesh is ``B^R``, where ``B`` defaults to 2.
"""
struct InherentDiscreteGrid{d,B} <: Grid{d,B}
    R::Int
    origin::NTuple{d,Int}
end

grid_min(grid::InherentDiscreteGrid) = grid.origin
grid_step(grid::InherentDiscreteGrid{d}) where {d} = ntuple(i -> 1, d)

"""
Create a grid for inherently discrete data with origin at 1
"""
InherentDiscreteGrid{d,B}(R::Int) where {d,B} = InherentDiscreteGrid{d,B}(R, ntuple(i -> 1, d))

"""
Create a grid for inherently discrete data with origin at 1 with a default ``B`` of 2
"""
InherentDiscreteGrid{d}(R::Int) where d = InherentDiscreteGrid{d,2}(R)

"""
Create a grid for inherently discrete data with a specified origin with a default ``B`` of 2
"""
InherentDiscreteGrid{d}(R::Int, origin::NTuple{d,Int}) where d = InherentDiscreteGrid{d,2}(R, origin)

"""
Create a grid for inherently discrete data with origin at 1
"""
InherentDiscreteGrid{d,B}(R::Int, origin::Int) where {d,B} = InherentDiscreteGrid{d,B}(R,  ntuple(i -> origin, d))

"""
Create a grid for inherently discrete data with origin at 1 with a default ``B`` of 2
"""
InherentDiscreteGrid{d}(R::Int, origin::Int) where d = InherentDiscreteGrid{d,2}(R, origin)

"""
Convert a coordinate in the original coordinate system to the corresponding grid index
"""
function origcoord_to_grididx(g::InherentDiscreteGrid, coordinate::Union{Int,NTuple{N,Int}}) where {N}
    return coordinate .- grid_min(g) .+ 1
end


@doc raw"""
The DiscretizedGrid struct represents a grid for discretized continuous data.
This is used for data that is originally continuous,
but has been discretized for computational purposes.
The grid contains values at specific, equally spaced points, which represent discrete 
approximations of the original continuous data. 
"""
struct DiscretizedGrid{d,B} <: Grid{d,B}
    R::Int
    grid_min::NTuple{d,Float64}
    grid_max::NTuple{d,Float64}
end


grid_min(g::DiscretizedGrid) = g.grid_min
grid_max(g::DiscretizedGrid) = g.grid_max
grid_step(g::DiscretizedGrid{d,B}) where {d,B} = (g.grid_max .- g.grid_min) ./ (B^g.R)

"""
Create a discrete grid for continuous data
"""
function DiscretizedGrid{d}(R::Int, grid_min, grid_max) where {d}
    return DiscretizedGrid{d,2}(R, grid_min, grid_max)
end

function DiscretizedGrid{d,B}(R::Int) where {d,B}
    return DiscretizedGrid{d,B}(R, ntuple(i -> 0.0, d), ntuple(i -> 1.0, d))
end

function DiscretizedGrid{d}(R::Int) where {d}
    return DiscretizedGrid{d,2}(R)
end


"""
Convert a coordinate in the original coordinate system to the corresponding grid index
"""
function origcoord_to_grididx(g::DiscretizedGrid, coordinate::NTuple{N,Float64}) where {N}
    all(grid_min(g) .<= coordinate .< grid_max(g)) ||
        error("Bound Error: $(coordinate), min=$(grid_min(g)), max=$(grid_max(g))")
    return ((coordinate .- grid_min(g)) ./ grid_step(g) .+ 1) .|> floor .|> Int
end

function origcoord_to_grididx(g::DiscretizedGrid{1,B}, coordinate::Float64) where {B}
    origcoord_to_grididx(g, (coordinate,))[1]
end
using TensorCrossInterpolation

struct HoneycombSite
    R::Vector{Int}
    c::Int

    function HoneycombSite(R::Vector{Int}, c::Int)
        if length(R) != 2 || (c != 1 && c != 2)
            throw(ArgumentError("Invalid site specification R = $R, c = $c"))
        end
        return new(R, c)
    end
end

function Base.isequal(l::HoneycombSite, r::HoneycombSite)
    return l.R == r.R && l.c == r.c
end

function Base.hash(s::HoneycombSite, h::UInt)
    return foldr(hash, [s.R, s.c, :HoneycombSite]; init=h)
end

function realspacecoordinates(s::HoneycombSite)
    A1::Vector{Float64} = [3 / 2, sqrt(3) / 2]
    A2::Vector{Float64} = [0, sqrt(3)]
    cshift::Vector{Float64} = s.c == 1 ? [0, 0] : [1, 0]
    return A1 * s.R[1] + A2 * s.R[2] + cshift
end

function neighbours(s::HoneycombSite)
    if s.c == 1
        return [
            HoneycombSite(s.R, 2),
            HoneycombSite(s.R + [-1, 0], 2),
            HoneycombSite(s.R + [-1, 1], 2)
        ]
    else
        return [
            HoneycombSite(s.R, 1),
            HoneycombSite(s.R + [1, 0], 1),
            HoneycombSite(s.R + [1, -1], 1)
        ]
    end
end

function nextneighbours(s::HoneycombSite)
    return [
        HoneycombSite(s.R + [1, 0], s.c),
        HoneycombSite(s.R + [1, -1], s.c),
        HoneycombSite(s.R + [-1, 0], s.c),
        HoneycombSite(s.R + [-1, 1], s.c),
        HoneycombSite(s.R + [0, 1], s.c),
        HoneycombSite(s.R + [0, -1], s.c)
    ]
end

function reducesite(s::HoneycombSite, Lx::Int, Ly::Int)
    xnew = mod(s.R[1], 2 * Lx)
    return HoneycombSite(
        [xnew, mod(s.R[2] - div(xnew - s.R[1], 2), Ly)],
        s.c
    )
end

function honeycomblattice(xmin::Integer, xmax::Integer, ymin::Integer, ymax::Integer)
    return TCI.IndexSet(
        [HoneycombSite([x, y], c) for c in 1:2, x in xmin:xmax, y in ymin:ymax][:]
    )
end

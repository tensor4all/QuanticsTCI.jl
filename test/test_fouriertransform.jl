import QuanticsTCI as QTCI
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG
import Random

# Legacy functions from QuanticsGrids <= v0.4
module LegacyQuanticsGrids

using StaticArrays

function quantics_to_index_fused(
    digitlist::AbstractVector{<:Integer};
    base::Integer=2,
    dims::Val{d}=Val(1),
)::NTuple{d,Int} where {d}
    R = length(digitlist)
    result = ones(MVector{d,Int})

    maximum(digitlist) <= base^d || error("maximum(digitlist) <= base^d")
    minimum(digitlist) >= 0 || error("minimum(digitlist) >= 0")

    for n = 1:R # from the least to most significant digit
        scale = base^(n - 1) # length scale
        tmp = digitlist[R-n+1] - 1
        for i = 1:d # in the order of 1st dim, 2nd dim, ...
            div_, rem_ = divrem(tmp, base)
            result[i] += rem_ * scale
            tmp = div_
        end
    end

    return tuple(result...)
end

function index_to_quantics!(digitlist, index::Integer; base::Integer=2)
    numdigits = length(digitlist)
    for i = 1:numdigits
        digitlist[i] = mod(index - 1, base^(numdigits - i + 1)) ÷ base^(numdigits - i) + 1
    end
    return digitlist
end

function index_to_quantics(index::Integer; numdigits=8, base::Integer=2)
    digitlist = Vector{Int}(undef, numdigits)
    return index_to_quantics!(digitlist, index; base=base)
end

end # module LegacyQuanticsGrids

@testset "Quantics Fourier Transform, R=$R" for R in [4, 16, 62]
    Random.seed!(23593243)

    r = 12
    coeffs = randn(ComplexF64, r)
    fm(x) = sum(coeffs .* cispi.(2 * (0:r-1) * x))
    fq(q) = fm((LegacyQuanticsGrids.quantics_to_index_fused(q)[1] - 1) / 2^big(R))

    qtci, = TCI.crossinterpolate2(ComplexF64, fq, fill(2, R); tolerance=1e-14)
    fouriertt = QTCI.quanticsfouriermpo(R; normalize=false) / 2^big(R)
    qtcif = TCI.contract(fouriertt, qtci)

    for i in 1:min(r, 2^big(R))
        q = LegacyQuanticsGrids.index_to_quantics(i, numdigits=R)
        @test qtcif(reverse(q)) ≈ coeffs[i]
    end

    for i in Int.(round.(range(r + 2, 2^big(R); length=100)))
        q = LegacyQuanticsGrids.index_to_quantics(i, numdigits=R)
        @test abs(qtcif(reverse(q))) < 1e-12
    end
end

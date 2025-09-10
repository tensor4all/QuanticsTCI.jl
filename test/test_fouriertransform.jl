import QuanticsTCI as QTCI
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG
import Random

"""
    function index_to_quantics!(digitlist, index::Integer; base::Integer=2)

* `digitlist`     base-b representation (1d vector)
* `base`           base for quantics (default: 2)
"""
function index_to_quantics!(digitlist, index::Integer; base::Integer=2)
    numdigits = length(digitlist)
    for i = 1:numdigits
        digitlist[i] = mod(index - 1, base^(numdigits - i + 1)) ÷ base^(numdigits - i) + 1
    end
    return digitlist
end

"""
    index_to_quantics(index::Integer; numdigits=8, base::Integer=2)

Does the same as [`index_to_quantics!`](@ref) but returns a new vector.
"""
function index_to_quantics(index::Integer; numdigits=8, base::Integer=2)
    digitlist = Vector{Int}(undef, numdigits)
    return index_to_quantics!(digitlist, index; base=base)
end

#@testset "Quantics Fourier Transform, R=$R" for R in [4, 16, 62]
@testset "Quantics Fourier Transform, R=$R" for R in [4, 16, 62]
    Random.seed!(23593243)

    grid = QG.DiscretizedGrid{1}(R, 0, 1)

    r = 12
    coeffs = randn(ComplexF64, r)
    fm(x) = sum(coeffs .* cispi.(2 * (0:r-1) * x))
    fq(q) = fm(QG.quantics_to_origcoord(grid, q)[1])

    qtci, = TCI.crossinterpolate2(ComplexF64, fq, fill(2, R); tolerance=1e-14)
    fouriertt = QTCI.quanticsfouriermpo(R; normalize=false) / 2^big(R)
    qtcif = TCI.contract(fouriertt, qtci)

    for i in 1:min(r, 2^big(R))
        q = index_to_quantics(i, numdigits=R)
        @test qtcif(reverse(q)) ≈ coeffs[i]
    end

    for i in Int.(round.(range(r+2, 2^big(R); length=100)))
        q = index_to_quantics(i, numdigits=R)
        @test abs(qtcif(reverse(q))) < 1e-12
    end
end

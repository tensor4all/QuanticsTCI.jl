import QuanticsTCI as QTCI
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

@testset "Quantics Fourier Transform" for R in [10, 30]
    function fq(q)
        m = QTCI.fourierimpl.quantics_to_int(BigInt, q)
        return cispi(2 * (m - 1) / big(2)^length(q))
    end

    qtci, = TCI.crossinterpolate2(ComplexF64, fq, fill(2, R); tolerance=1e-14)
    @info "QTCI rank" TCI.rank(qtci)
    fouriertt = QTCI.quanticsfouriertto(R; tolerance=1e-14)
    @info "FT rank" TCI.rank(fouriertt)
    qtcif = TCI.contract(qtci, fouriertt)
    @info "transformed QTCI rank" TCI.rank(qtcif)

    @test qtcif(ones(Int, R)) ≈ 0.0
    @test qtcif([2, ones(Int, R-1)...]) ≈ 2.0 ^ R
    @test qtcif([1, 2, ones(Int, R-2)...]) ≈ 0.0

    for i in range(4, 2^R, length=100)
        q = QG.index_to_quantics(Int(round(i)), numdigits=R)
        @test abs(qtcif(reverse(q))) < 1e-12
    end
end

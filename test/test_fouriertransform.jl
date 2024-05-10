import QuanticsTCI as QTCI
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG
import Random

@testset "Quantics Fourier Transform, R=$R" for R in [4, 16, 62]
    Random.seed!(23593243)

    r = 12
    coeffs = randn(ComplexF64, r)
    fm(x) = sum(coeffs .* cispi.(2 * (0:r-1) * x))
    fq(q) = fm((QG.quantics_to_index_fused(q)[1] - 1) / 2^big(R))

    qtci, = TCI.crossinterpolate2(ComplexF64, fq, fill(2, R); tolerance=1e-14)
    fouriertt = QTCI.quanticsfouriertto(R) / 2^big(R)
    qtcif = TCI.contract(qtci, fouriertt)

    for i in 1:min(r, 2^big(R))
        q = QG.index_to_quantics(i, numdigits=R)
        @test qtcif(reverse(q)) ≈ coeffs[i]
    end

    for i in Int.(round.(range(r+2, 2^big(R); length=100)))
        q = QG.index_to_quantics(i, numdigits=R)
        @test abs(qtcif(reverse(q))) < 1e-12
    end
end

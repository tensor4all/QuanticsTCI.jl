using Random
import QuanticsGrids as QG

@testset "quanticscrossinterpolate" begin
    f(x, y) = 0.1 * x^2 + 0.01 * y^3 - pi * x * y + 5
    xvals = range(-3, 2; length=32)
    yvals = range(-17, 12; length=32)
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals, yvals]; tolerance=1e-8)
    @test last(errors) < 1e-8

    for (i, x) in enumerate(xvals)
        for (j, y) in enumerate(yvals)
            @test f(x, y) ≈ qtt(i, j)
        end
    end
end

@testset "quanticscrossinterpolate, 1d overload" begin
    f(x) = 0.1 * x^2 - pi * x + 2
    g(x) = f(x[1])
    xvals = range(-3, 2; length=128)
    Random.seed!(1234)
    qttf, ranksf, errorsf = quanticscrossinterpolate(Float64, f, xvals; tolerance=1e-8)
    Random.seed!(1234)
    qttg, ranksg, errorsg = quanticscrossinterpolate(Float64, g, [xvals]; tolerance=1e-8)
    @test last(errorsf) < 1e-8
    @test last(errorsg) < 1e-8
    @test ranksf == ranksg
    @test errorsf == errorsg

    for (i, x) in enumerate(xvals)
        @test f(x) == g([x])
        @test f(x) ≈ qttf(i)
        @test g([x]) ≈ qttg([i])
    end
end

@testset "quanticscrossinterpolate with DiscretizedGrid" begin
    R = 5
    f(x, y) = 0.1 * x^2 + 0.01 * y^3 - pi * x * y + 5
    grid = QG.DiscretizedGrid{2}(
        R,
        (-3, -17),
        (2, 12)
    )
    Random.seed!(1234)
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f, grid; tolerance=1e-8)
    @test last(errors) < 1e-8

    for i in 1:2^R
        for j in 1:2^R
            @test f(QG.grididx_to_origcoord(grid, (i, j))...) ≈ qtt(i, j)
        end
    end
end

@testset "quanticscrossinterpolate with InherentDiscreteGrid" for unfoldingscheme in [
    QG.UnfoldingSchemes.interleaved,
    QG.UnfoldingSchemes.fused
]
    R = 3
    Random.seed!(1234)
    A = rand(2^R, 2^R, 2^R)
    grid = QG.InherentDiscreteGrid{3}(R; unfoldingscheme)
    qtt, ranks, errors = quanticscrossinterpolate(
        Float64, (i...) -> A[i...], grid; tolerance=1e-8)
    @test last(errors) < 1e-8
    for i in CartesianIndices(size(A))
        @test A[i] ≈ qtt(Tuple(i))
    end
end

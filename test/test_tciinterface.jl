@testset "quanticscrossinterpolate" begin
    f(x, y) = exp(-x - 2y)
    xvals = range(-3, 2; length=32)
    yvals = range(-17, 12; length=32)
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals, yvals]; tolerance=1e-8)
    @test rank(qtt.tt) == 2
    @test last(errors) < 1e-8

    for (i, x) in enumerate(xvals)
        for (j, y) in enumerate(yvals)
            @test f(x, y) ≈ qtt(i, j)
        end
    end
end

@testset "quanticscrossinterpolate, 1d overload" begin
    f(x) = exp(-3x)
    g(x) = f(x[1])
    xvals = range(-3, 2; length=128)
    qttf, ranksf, errorsf = quanticscrossinterpolate(Float64, f, xvals; tolerance=1e-8)
    qttg, ranksg, errorsg = quanticscrossinterpolate(Float64, g, [xvals]; tolerance=1e-8)
    @test rank(qttf.tt) == 2
    @test last(errorsf) < 1e-8
    @test rank(qttg.tt) == 2
    @test last(errorsg) < 1e-8
    @test ranksf == ranksg
    @test errorsf == errorsg

    for (i, x) in enumerate(xvals)
        @test f(x) == g([x])
        @test f(x) ≈ qttf(i)
        @test g([x]) ≈ qttg([i])
    end
end

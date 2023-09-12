@testset "quanticscrossinterpolate" begin
    f(x) = exp(-maximum(x))
    xvals = range(-3, 2; length=256)
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals]; tolerance=1e-8)
    @test rank(qtt.tt) == 2
    @test last(errors) < 1e-8

    for (i, x) in enumerate(xvals)
        @test f(x) ≈ qtt(i)
    end

    qtt2, ranks2, errors2 = quanticscrossinterpolate(Float64, f, xvals; tolerance=1e-8)
    @test rank(qtt2.tt) == 2
    @test last(errors) < 1e-8
    @test ranks2 ≈ ranks
    @test errors2 ≈ errors

    for (i, x) in enumerate(xvals)
        @test f(x) ≈ qtt(i)
    end
end

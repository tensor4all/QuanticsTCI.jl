using Random
import QuanticsGrids as QG

# :serial and :meander were added to QuanticsGrids in v0.8.0. The compat entry of
# QuanticsTCI still allows older releases, so probe once which schemes the
# resolved QuanticsGrids understands instead of hard-coding a version check.
function _supportedunfoldingschemes()
    schemes = Symbol[]
    for scheme in (:interleaved, :fused, :serial, :meander)
        try
            QG.InherentDiscreteGrid{2}(1; unfoldingscheme=scheme)
            push!(schemes, scheme)
        catch err
            err isa ArgumentError || rethrow()
        end
    end
    return schemes
end

UNFOLDINGSCHEMES = _supportedunfoldingschemes()

@testset "quanticscrossinterpolate" for unfoldingscheme in UNFOLDINGSCHEMES
    f(x, y) = 0.1 * x^2 + 0.01 * y^3 - pi * x * y + 5
    xvals = range(-3, 2; length=32)
    yvals = range(-17, 12; length=32)
    qtt, ranks, errors = quanticscrossinterpolate(
        Float64, f, [xvals, yvals]; unfoldingscheme, tolerance=1e-8)
    @test last(errors) < 1e-8

    cache = cachedata(qtt)
    for (k, v) in cache
        @test v ≈ f(k...)
    end

    for (i, x) in enumerate(xvals)
        for (j, y) in enumerate(yvals)
            @test f(x, y) ≈ qtt(i, j)
        end
    end

    @test sum(qtt) ≈ sum(f.(xvals, transpose(yvals)))
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

@testset "quanticscrossinterpolate with DiscretizedGrid" for unfoldingscheme in UNFOLDINGSCHEMES
    R = 5
    f(x, y) = 0.1 * x^2 + 0.01 * y^3 - pi * x * y + 5
    grid = QG.DiscretizedGrid{2}(
        R,
        (-3, -17),
        (2, 12);
        unfoldingscheme
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

@testset "quanticscrossinterpolate with InherentDiscreteGrid" for unfoldingscheme in UNFOLDINGSCHEMES
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

    qtt, ranks, errors = quanticscrossinterpolate(
        Float64, (i...) -> A[i...], size(A); unfoldingscheme, tolerance=1e-8)
    @test last(errors) < 1e-8
    for i in CartesianIndices(size(A))
        @test A[i] ≈ qtt(Tuple(i))
    end

    qtt, ranks, errors = quanticscrossinterpolate(
        A; unfoldingscheme, tolerance=1e-8)
    @test last(errors) < 1e-8
    for i in CartesianIndices(size(A))
        @test A[i] ≈ qtt(Tuple(i))
    end
end

@testset "quanticscrossinterpolate for integrals" begin
    R = 40
    xgrid = QG.DiscretizedGrid{1}(R, 0, 1)
    F(x) = sin(1/(x^2 + 0.01))
    f(x) = -2*x * cos(1/(x^2 + 0.01)) / (x^2 + 0.01)^2
    tci, ranks, errors = quanticscrossinterpolate(Float64, f, xgrid; tolerance=1e-13)
    @test sum(tci) * QG.grid_step(xgrid) ≈ F(1) - F(0)
    @test integral(tci) ≈ F(1) - F(0)
end

# The three "digit-level" unfoldingschemes must all give the same quantics tensor
# train for the same function on the same grid: they only differ in the ordering
# of the sites, which is handled by QuanticsGrids.
@testset "unfoldingschemes :interleaved, :serial and :meander agree" begin
    schemes = intersect([:interleaved, :serial, :meander], UNFOLDINGSCHEMES)
    if length(schemes) < 3
        @warn "Skipping: the resolved QuanticsGrids does not support $(setdiff([:interleaved, :serial, :meander], schemes))"
    else
        R = 5
        tolerance = 1e-10
        f(x, y) = exp(-0.1 * x^2) * cos(x + 0.5 * y) + 0.3 * y
        grids = map(schemes) do scheme
            QG.DiscretizedGrid{2}(R, (-3, -17), (2, 12); unfoldingscheme=scheme)
        end
        qtcis = map(grids) do grid
            Random.seed!(1234)
            qtt, ranks, errors = quanticscrossinterpolate(
                Float64, f, grid; tolerance=tolerance)
            @test last(errors) < tolerance
            @test maximum(abs(qtt(i, j) - f(QG.grididx_to_origcoord(grid, (i, j))...))
                          for i in 1:2^R, j in 1:2^R) < 1e-8
            qtt
        end
        for i in 1:2^R, j in 1:2^R
            @test qtcis[1](i, j) ≈ qtcis[2](i, j) atol = 1e-10
            @test qtcis[1](i, j) ≈ qtcis[3](i, j) atol = 1e-10
        end
        @test sum(qtcis[1]) ≈ sum(qtcis[2]) atol = 1e-8
        @test sum(qtcis[1]) ≈ sum(qtcis[3]) atol = 1e-8
    end
end

# Verify that the site ordering seen by the TCI is the one QuanticsGrids
# documents for each scheme, i.e. that QuanticsTCI does not reorder anything:
#   :interleaved  x_1 y_1 x_2 y_2 x_3 y_3
#   :serial       x_1 x_2 x_3 y_1 y_2 y_3
#   :meander      x_3 x_2 x_1 y_1 y_2 y_3  (odd-numbered variables reversed)
@testset "digit order of :serial and :meander" begin
    R = 3
    # (variable index, digit number), most significant digit of a variable first
    expectedsites = Dict(
        :interleaved => [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)],
        :serial => [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)],
        :meander => [(1, 3), (1, 2), (1, 1), (2, 1), (2, 2), (2, 3)],
    )
    # 1-based digit `k` (1 = most significant) of the 0-based integer `idx - 1`
    digit(idx, k) = (((idx - 1) >> (R - k)) & 1) + 1

    for scheme in intersect(collect(keys(expectedsites)), UNFOLDINGSCHEMES)
        grid = QG.InherentDiscreteGrid{2}(R; unfoldingscheme=scheme)
        @test QG.localdimensions(grid) == fill(2, 2 * R)
        for (i, j) in ((1, 1), (2, 7), (5, 2), (8, 8), (4, 6))
            q = QG.grididx_to_quantics(grid, (i, j))
            @test length(q) == 2 * R
            for (site, (var, k)) in enumerate(expectedsites[scheme])
                @test q[site] == digit(var == 1 ? i : j, k)
            end
            @test QG.quantics_to_grididx(grid, q) == (i, j)
        end
    end

    # The TCI evaluates exactly the quantics indices that QuanticsGrids produces.
    if :serial in UNFOLDINGSCHEMES && :meander in UNFOLDINGSCHEMES
        R = 4
        f(x, y) = x + 10 * y
        for scheme in (:serial, :meander)
            grid = QG.InherentDiscreteGrid{2}(R; unfoldingscheme=scheme)
            qtt, ranks, errors = quanticscrossinterpolate(
                Float64, f, grid; tolerance=1e-12)
            @test last(errors) < 1e-12
            for i in 1:2^R, j in 1:2^R
                x, y = QG.grididx_to_origcoord(grid, (i, j))
                @test qtt(i, j) ≈ f(x, y)
            end
        end
    end
end

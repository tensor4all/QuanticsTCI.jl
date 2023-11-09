@testset "quantics representation" begin
    @testset "quantics to index, 1d" begin
        B = 2
        d = 1
        @test quantics_to_index_fused(Val(B), Val(d), [1, 1, 1, 1]) == (1,)
        @test quantics_to_index_fused(Val(B), Val(d), [1, 1, 1, 2]) == (2,)
        @test quantics_to_index_fused(Val(B), Val(d), [1, 1, 2, 1]) == (3,)
        @test quantics_to_index_fused(Val(B), Val(d), [1, 1, 2, 2]) == (4,)
        @test quantics_to_index_fused(Val(B), Val(d), [2, 1, 1, 1]) == (9,)
    end

    @testset "quantics to index, 1d (general power base)" for B in [2, 4]
        d = 1
        R = 4 # number of digits

        index_reconst = Int[]
        for index in 1:B^R
            digitlist_ = QuanticsTCI.index_to_quantics(Val(B), index; numdigits=R)
            push!(index_reconst, only(quantics_to_index_fused(Val(B), Val(d), digitlist_)))
        end

        @test collect(1:B^R) == index_reconst
    end

    @testset "quantics to index, 2d, base=3" begin
        base = 3
        dim = 2

        # X_i = Fused quantics index at i (1 <= i <= R)
        # x_i = quantics index for the first variable at i (1 <= i <= R)
        # y_i = quantics index for the second variable at i (1 <= i <= R)
        #
        # X_i = (x_i-1) + (base) * (y_i-1) + 1 (column major)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 1]) == (1, 1)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 2]) == (2, 1)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 3]) == (3, 1)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 4]) == (1, 2)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 5]) == (2, 2)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 6]) == (3, 2)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 7]) == (1, 3)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 8]) == (2, 3)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [1, 9]) == (3, 3)
        @test QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), [2, 1]) == (4, 1)
    end

    @testset "quantics back-and-forth, 2d" begin
        base = 2
        dim = 2
        R = 2
        
        for j in 1:base^R, i in 1:base^R
            index = (i, j)
            digitlist = Vector{Int}(undef, R)
            QuanticsTCI.index_to_quantics_fused!(Val(base), digitlist, index)
            index_reconst = QuanticsTCI.quantics_to_index_fused(Val(base), Val(dim), digitlist)
            @test index == index_reconst
        end
    end

    #==
    @testset "index to quantics, 1d" begin
        @test [1, 1, 1, 1] == index_to_quantics([1], 4)
        @test [1, 1, 1, 2] == index_to_quantics([2], 4)
        @test [1, 1, 2, 1] == index_to_quantics([3], 4)
        @test [1, 1, 2, 2] == index_to_quantics([4], 4)
        @test [2, 1, 1, 1] == index_to_quantics([9], 4)
    end

    @testset "quantics back-and-forth, 1d" begin
        n = 4
        npoints = 2^n
        for i in 1:npoints
            q = index_to_quantics(i, n)
            @test [((i - 1) & 2^(n - j)) != 0 for j in 1:n] == q .- 1
            @test quantics_to_index(q) == i
        end
    end

    @testset "quantics to index, 3d" begin
        @test quantics_to_index([1 1 1 1], 3) == [1, 1, 1]
        @test quantics_to_index([1 1 1 2], 3) == [2, 1, 1]
        @test quantics_to_index([1 1 2 1], 3) == [3, 1, 1]
        @test quantics_to_index([1 1 2 2], 3) == [4, 1, 1]
        @test quantics_to_index([2 1 1 1], 3) == [9, 1, 1]
        @test quantics_to_index([1 1 2 3], 3) == [3, 2, 1]
        @test quantics_to_index([1 1 2 4], 3) == [4, 2, 1]
        @test quantics_to_index([1 1 2 5], 3) == [3, 1, 2]
        @test quantics_to_index([1 1 2 8], 3) == [4, 2, 2]
    end

    @testset "index to quantics, 3d" begin
        @test [1, 1, 1, 1] == index_to_quantics([1, 1, 1], 4)
        @test [1, 1, 1, 2] == index_to_quantics([2, 1, 1], 4)
        @test [1, 1, 2, 1] == index_to_quantics([3, 1, 1], 4)
        @test [1, 1, 2, 2] == index_to_quantics([4, 1, 1], 4)
        @test [2, 1, 1, 1] == index_to_quantics([9, 1, 1], 4)
        @test [1, 1, 2, 3] == index_to_quantics([3, 2, 1], 4)
        @test [1, 1, 2, 4] == index_to_quantics([4, 2, 1], 4)
        @test [1, 1, 2, 5] == index_to_quantics([3, 1, 2], 4)
        @test [1, 1, 2, 8] == index_to_quantics([4, 2, 2], 4)
    end

    @testset "quantics back-and-forth, 3d" begin
        n = 4
        npoints = 2^n
        for x in 1:npoints, y in 1:npoints, z in 1:npoints
            q = index_to_quantics([x, y, z], n)
            @test q == index_to_quantics_fused([x, y, z], n)
            @test (
                1 .* [((x - 1) & 2^(n - j)) != 0 for j in 1:n] .+
                2 .* [((y - 1) & 2^(n - j)) != 0 for j in 1:n] .+
                4 .* [((z - 1) & 2^(n - j)) != 0 for j in 1:n]
            ) == q .- 1
            @test quantics_to_index(q, 3) == [x, y, z]
        end
    end

    @testset "split dimensions" begin
        @test split_dimensions([1, 1, 1, 1], 1) == [[1, 1, 1, 1]]
        @test split_dimensions([1, 1, 1, 1], 3) == [[1, 1, 1, 1] for i in 1:3]
        @test split_dimensions([2, 2, 2, 2], 3) == [[2, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1]]
        @test split_dimensions([3, 3, 3, 3], 3) == [[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]]
        @test split_dimensions([5, 5, 5, 5], 3) == [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]]
        @test split_dimensions([1, 2, 4, 8], 3) == [[1, 2, 2, 2], [1, 1, 2, 2], [1, 1, 1, 2]]
    end

    @testset "merge dimensions" begin
        @test [1, 1, 1, 1] == merge_dimensions([1, 1, 1, 1])
        @test [1, 1, 1, 1] == merge_dimensions([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
        @test [2, 2, 2, 2] == merge_dimensions([2, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1])
        @test [3, 3, 3, 3] == merge_dimensions([1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1])
        @test [5, 5, 5, 5] == merge_dimensions([1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2])
        @test [1, 2, 4, 8] == merge_dimensions([1, 2, 2, 2], [1, 1, 2, 2], [1, 1, 1, 2])
    end

    @testset "merge is inverse of split dimensions" begin
        d = 3
        np = 2^d
        for i1 in 1:np, i2 in 1:np, i3 in 1:np, i4 in 1:np
            q = [i1, i2, i3, i4]
            @test q == merge_dimensions(split_dimensions(q, d)...)
        end
    end

    @testset "split is inverse of merge dimensions" begin
        n = 4
        np = 2^n
        for i1 in 1:np, i2 in 1:np
            q1 = index_to_quantics(i1, n)
            q2 = index_to_quantics(i2, n)
            @test [q1, q2] == split_dimensions(merge_dimensions(q1, q2), 2)
        end
    end

    @testset "interleave dimensions" begin
        @test [1, 1, 1, 1] == interleave_dimensions([1, 1, 1, 1])
        @test [1, 1, 1, 1, 1, 1, 1, 1] == interleave_dimensions([1, 1, 1, 1], [1, 1, 1, 1])
        @test [1, 2, 1, 3, 1, 4, 1, 5] == interleave_dimensions([1, 1, 1, 1], [2, 3, 4, 5])
        @test [1, 2, 11, 1, 3, 12, 1, 4, 13, 1, 5, 14] == interleave_dimensions([1, 1, 1, 1], [2, 3, 4, 5], [11, 12, 13, 14])
    end

    @testset "deinterleave dimensions" begin
        @test deinterleave_dimensions([1, 1, 1, 1], 1) == [[1, 1, 1, 1]]
        @test deinterleave_dimensions([1, 1, 1, 1], 2) == [[1, 1], [1, 1]]
        @test deinterleave_dimensions([1, 2, 1, 3, 1, 4, 1, 5], 2) == [[1, 1, 1, 1], [2, 3, 4, 5]]
        @test deinterleave_dimensions([1, 2, 11, 1, 3, 12, 1, 4, 13, 1, 5, 14], 3) == [[1, 1, 1, 1], [2, 3, 4, 5], [11, 12, 13, 14]]
    end

    @testset "deinterleave is inverse of interleave dimension" begin
        n = 4
        np = 2^n
        for i1 in 1:np, i2 in 1:np
            q1 = index_to_quantics(i1, n)
            q2 = index_to_quantics(i2, n)
            @test [q1, q2] == deinterleave_dimensions(interleave_dimensions(q1, q2), 2)
        end
    end

    @testset "interleave is inverse of deinterleave dimensions" begin
        d = 3
        np = 2^d
        for i1 in 1:np, i2 in 1:np, i3 in 1:np, i4 in 1:np
            q = [i1, i2, i3, i4]
            @test q == interleave_dimensions(deinterleave_dimensions(q, 2)...)
        end
    end

    @testset "fuse and interleave function wrappers are consistent" begin
        n = 4
        np = 2^n
        for i1 in 1:np, i2 in 1:np
            q1 = index_to_quantics(i1, n)
            q2 = index_to_quantics(i2, n)
            @test fuse_dimensions(q1, q2) == index_to_quantics_fused([i1, i2], n)
            @test interleave_dimensions(q1, q2) == index_to_quantics_interleaved([i1, i2], n)
        end
    end
    end

    @testset "quantics function wrappers" begin
    f(u) = u

    qf = QuanticsFunction{Int}(f)
    for i in 1:10
        @test qf(index_to_quantics(i, 4)) == i
    end

    qfinterleaved = QuanticsFunctionInterleaved{Vector{Int}}(f, 3)
    for i in 1:10
        for j in 1:10
            for k in 1:10
                qindex = interleave_dimensions(index_to_quantics.([i, j, k], 4)...)
                @test qfinterleaved(qindex) == [i, j, k]
            end
        end
    end

    qffused = QuanticsFunctionFused{Vector{Int}}(f, 3)
    for i in 1:10
        for j in 1:10
            for k in 1:10
                qindex = merge_dimensions(index_to_quantics.([i, j, k], 4)...)
                @test qffused(qindex) == [i, j, k]
            end
        end
    end
    ==#
end

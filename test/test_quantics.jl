@testset "quantics representation" begin
    @testset "quantics to index, 1d" begin
        @test quantics_to_index([1 1 1 1]) == [1]
        @test quantics_to_index([2 1 1 1]) == [2]
        @test quantics_to_index([1 2 1 1]) == [3]
        @test quantics_to_index([2 2 1 1]) == [4]
        @test quantics_to_index([1 1 1 2]) == [9]
    end

    @testset "index to quantics, 1d" begin
        @test [1, 1, 1, 1] == index_to_quantics([1], 4)
        @test [2, 1, 1, 1] == index_to_quantics([2], 4)
        @test [1, 2, 1, 1] == index_to_quantics([3], 4)
        @test [2, 2, 1, 1] == index_to_quantics([4], 4)
        @test [1, 1, 1, 2] == index_to_quantics([9], 4)
    end

    @testset "quantics back-and-forth, 1d" begin
        n = 4
        npoints = 2^n
        for i in 1:npoints
            q = index_to_quantics(i, n)
            @test [((i - 1) & 2^(j - 1)) != 0 for j in 1:n] == q .- 1
            @test quantics_to_index(q) == [i]
        end
    end

    @testset "quantics to index, 3d" begin
        @test quantics_to_index([1 1 1 1]; d=3) == [1, 1, 1]
        @test quantics_to_index([2 1 1 1]; d=3) == [2, 1, 1]
        @test quantics_to_index([1 2 1 1]; d=3) == [3, 1, 1]
        @test quantics_to_index([2 2 1 1]; d=3) == [4, 1, 1]
        @test quantics_to_index([1 1 1 2]; d=3) == [9, 1, 1]
        @test quantics_to_index([3 2 1 1]; d=3) == [3, 2, 1]
        @test quantics_to_index([4 2 1 1]; d=3) == [4, 2, 1]
        @test quantics_to_index([5 2 1 1]; d=3) == [3, 1, 2]
        @test quantics_to_index([8 2 1 1]; d=3) == [4, 2, 2]
    end

    @testset "index to quantics, 3d" begin
        @test [1, 1, 1, 1] == index_to_quantics([1, 1, 1], 4)
        @test [2, 1, 1, 1] == index_to_quantics([2, 1, 1], 4)
        @test [1, 2, 1, 1] == index_to_quantics([3, 1, 1], 4)
        @test [2, 2, 1, 1] == index_to_quantics([4, 1, 1], 4)
        @test [1, 1, 1, 2] == index_to_quantics([9, 1, 1], 4)
        @test [3, 2, 1, 1] == index_to_quantics([3, 2, 1], 4)
        @test [4, 2, 1, 1] == index_to_quantics([4, 2, 1], 4)
        @test [5, 2, 1, 1] == index_to_quantics([3, 1, 2], 4)
        @test [8, 2, 1, 1] == index_to_quantics([4, 2, 2], 4)
    end

    @testset "quantics back-and-forth, 3d" begin
        n = 4
        npoints = 2^n
        for x in 1:npoints, y in 1:npoints, z in 1:npoints
            q = index_to_quantics([x, y, z], n)
            @test (
                1 .* [((x - 1) & 2^(j - 1)) != 0 for j in 1:n] .+
                2 .* [((y - 1) & 2^(j - 1)) != 0 for j in 1:n] .+
                4 .* [((z - 1) & 2^(j - 1)) != 0 for j in 1:n]
            ) == q .- 1
            @test quantics_to_index(q; d=3) == [x, y, z]
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
end

using QuanticsTCI
using Test

@testset "QuanticsTCI.jl" begin
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
end

@testset "mps util" begin
    @testset "one mps evaluation" begin
        m = 5
        mps = MPS(m)
        linkindices = [Index(2, "link") for i in 1:(m-1)]
        mps[1] = ITensor(
            ones(2, 2),
            Index(2, "site"),
            linkindices[1]
        )
        for i in 2:(m-1)
            mps[i] = ITensor(
                ones(2, 2, 2),
                linkindices[i-1],
                Index(2, "site"),
                linkindices[i]
            )
        end
        mps[m] = ITensor(
            ones(2, 2),
            linkindices[m-1],
            Index(2, "site")
        )

        for i in 1:(2^m)
            q = index_to_quantics(i, m)
            @test evaluate_mps(mps, siteinds(mps), q) == 2^(m - 1)
            @test evaluate_mps(mps, collect(zip(siteinds(mps), q))) == 2^(m - 1)
        end
    end

    @testset "delta mps evaluation" begin
        m = 5
        mps = MPS(m)
        linkindices = [Index(2, "link") for i in 1:(m-1)]
        mps[1] = ITensor(
            [i == j for i in 0:1, j in 0:1],
            Index(2, "site"),
            linkindices[1]
        )
        for i in 2:(m-1)
            mps[i] = ITensor(
                [i == j && j == k for i in 0:1, j in 0:1, k in 0:1],
                linkindices[i-1],
                Index(2, "site"),
                linkindices[i]
            )
        end
        mps[m] = ITensor(
            [i == j for i in 0:1, j in 0:1],
            linkindices[m-1],
            Index(2, "site")
        )

        for i in 1:(2^m)
            q = index_to_quantics(i, m)
            @test evaluate_mps(mps, siteinds(mps), q) == all(q .== first(q))
            @test (
                evaluate_mps(mps, collect(zip(siteinds(mps), q))) ==
                all(q .== first(q))
            )
        end
    end
end
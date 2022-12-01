@testset "MatrixCI" begin
    @testset "Matrix util" begin
        A = [
            0.262819 0.740968 0.505743
            0.422301 0.831443 0.32687
            0.439065 0.426132 0.453675
            0.128233 0.0490983 0.902257
            0.371653 0.810275 0.75838
        ]
        eye(n) = Matrix(LinearAlgebra.I, n, n)
        @test A ≈ QuanticsTCI.MatrixCIutil.AtimesBinv(A, eye(3))
        @test A ≈ QuanticsTCI.MatrixCIutil.AinvtimesB(eye(5), A)

        B = [
            0.852891 0.945401 0.585575
            0.800289 0.478038 0.661408
            0.685688 0.619311 0.309872
        ]
        C = [
            0.304463 0.399473 0.767147 0.337228 0.86603
            0.147815 0.508933 0.794015 0.326105 0.8079
            0.665499 0.0571589 0.766872 0.167927 0.028576
            0.411886 0.397681 0.473644 0.527007 0.4264
            0.244107 0.0669144 0.347337 0.947754 0.76624
        ]

        @test eye(3) ≈ QuanticsTCI.MatrixCIutil.AtimesBinv(B, B)
        @test eye(3) ≈ QuanticsTCI.MatrixCIutil.AinvtimesB(B, B)
        @test eye(5) ≈ QuanticsTCI.MatrixCIutil.AtimesBinv(C, C)
        @test eye(5) ≈ QuanticsTCI.MatrixCIutil.AinvtimesB(C, C)

        @test A * inv(B) ≈ QuanticsTCI.MatrixCIutil.AtimesBinv(A, B)
        @test inv(C) * A ≈ QuanticsTCI.MatrixCIutil.AinvtimesB(C, A)
    end

    @testset "Empty constructor" begin
        ci = QuanticsTCI.matrix_cross_interpolation{Float64}(10, 25)

        @test ci.row_indices == []
        @test ci.col_indices == []
        @test ci.pivot_cols == zeros(10, 0)
        @test ci.pivot_rows == zeros(0, 25)
        @test QuanticsTCI.n_rows(ci) == 10
        @test QuanticsTCI.n_cols(ci) == 25
    end

    @testset "Full constructor" begin
        A = [
            0.735188   0.718229   0.206528  0.89223   0.23432;
            0.58692    0.383284   0.906576  0.3389    0.24915;
            0.0866507  0.812134   0.683979  0.798798  0.63418;
            0.694491   0.585013   0.623725  0.25272   0.72730;
            0.100076   0.248325   0.770408  0.342828  0.080717;
            0.748823   0.653965   0.47961   0.909719  0.037413;
            0.902325   0.743668   0.193464  0.380086  0.91558;
            0.0614368  0.0709293  0.343843  0.197515  0.45067;
        ]

        row_indices = [8, 2, 3]
        col_indices = [1, 5, 4]

        ci = QuanticsTCI.matrix_cross_interpolation(
            row_indices, col_indices,
            A[:, col_indices], A[row_indices, :]
        )

        @test ci.row_indices == row_indices
        @test ci.col_indices == col_indices
        @test ci.pivot_cols == A[:, col_indices]
        @test ci.pivot_rows == A[row_indices, :]
        @test QuanticsTCI.n_rows(ci) == 8
        @test QuanticsTCI.n_cols(ci) == 5

        Apivot = A[row_indices, col_indices]
        @test QuanticsTCI.pivot_matrix(ci) == Apivot
        @test QuanticsTCI.left_matrix(ci) ≈ A[:, col_indices] * inv(Apivot)
        @test QuanticsTCI.right_matrix(ci) ≈ inv(Apivot) * A[row_indices, :]

        @test QuanticsTCI.avail_rows(ci) == [1, 4, 5, 6, 7]
        @test QuanticsTCI.avail_cols(ci) == [2, 3]
        
        for i in row_indices, j in col_indices
            @test QuanticsTCI.eval(ci, i, j) ≈ A[i, j]
            ci[i, j] ≈ A[i, j]
        end

        for i in row_indices
            @test QuanticsTCI.row(ci, i)[col_indices] ≈ A[i, col_indices]
            @test ci[i, col_indices] ≈ A[i, col_indices]
        end

        for j in col_indices
            @test QuanticsTCI.col(ci, j)[row_indices] ≈ A[row_indices, j]
            @test ci[row_indices, j] ≈ A[row_indices, j]
        end

        @test QuanticsTCI.submatrix(ci, row_indices, col_indices) ≈ 
            A[row_indices, col_indices]
        @test ci[row_indices, col_indices] ≈ A[row_indices, col_indices]
        @test QuanticsTCI.matrix(ci)[row_indices, col_indices] ≈ A[row_indices, col_indices]
    end
end
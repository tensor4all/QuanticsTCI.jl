using QuanticsTCI
import TensorCrossInterpolation as TCI

# Number of bits
R = 8

# Replace with your dataset
grid = range(-pi, pi; length=2^R+1)[1:end-1] # exclude the end point
dataset = [cos(x) + cos(y) + cos(z) for x in grid, y in grid, z in grid]

# Perform QTCI
tolerance = 1e-5
qtt, ranks, errors = quanticscrossinterpolate(
    dataset, tolerance=tolerance, unfoldingscheme=:fused)

# Check error
qttdataset = [qtt([i, j, k]) for i in axes(grid, 1), j in axes(grid, 1), k in axes(grid, 1)]
error = abs.(qttdataset .- dataset)
println(
    "Quantics TCI compression of the dataset with tolerance $tolerance has " *
    "link dimensions $(TCI.linkdims(qtt.tci)), for a max error of $(maximum(error))."
)

import TensorCrossInterpolation as TCI

# Replace this line with the dataset to be tested for compressibility.
grid = range(-pi, pi; length=200)
dataset = [cos(x) + cos(y) + cos(z) for x in grid, y in grid, z in grid]

# Construct TCI
tolerance = 1e-5
tt, ranks, errors = TCI.crossinterpolate2(
    Float64, i -> dataset[i...], collect(size(dataset)), tolerance=tolerance)

# Check error
ttdataset = [tt([i, j, k]) for i in axes(grid, 1), j in axes(grid, 1), k in axes(grid, 1)]
errors = abs.(ttdataset .- dataset)
println(
    "TCI of the dataset with tolerance $tolerance has link dimensions $(TCI.linkdims(tt)), "
    * "for a max error of $(maximum(errors))."
)

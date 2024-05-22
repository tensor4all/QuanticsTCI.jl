import TensorCrossInterpolation as TCI
import Random
import QuanticsGrids as QD
#using PythonPlot: pyplot as plt

# Number of bits
R = 4
tol = 1e-4

# f(q) = 1 if q = (1, 1, ..., 1) or q = (2, 2, ..., 2), 0 otherwise
f(q) = (all(q .== 1) || all(q .== 2)) ? 1.0 : 0.0

localdims = fill(2, R)

# Perform TCI with an initial pivot at (1, 1, ..., 1)
firstpivot = ones(Int, R)
tci, ranks, errors = TCI.crossinterpolate2(
    Float64,
    f,
    localdims,
    [firstpivot];
    tolerance=tol,
    nsearchglobalpivot=0 # Disable automatic global pivot search
)

# TCI fails to capture the function at (2, 2, ..., 2)
globalpivot = fill(2, R)
@assert isapprox(TCI.evaluate(tci, globalpivot), 0.0)

# Add (2, 2, ..., 2) as a global pivot
tci_globalpivot = deepcopy(tci)
TCI.addglobalpivots2sitesweep!(
    tci_globalpivot, f, [globalpivot],
    tolerance=tol
)
@assert isapprox(TCI.evaluate(tci_globalpivot, globalpivot), 1.0)

# Plot the function and the TCI reconstructions
grid = QD.InherentDiscreteGrid{1}(R)
ref = [f(QD.grididx_to_quantics(grid, i)) for i in 1:2^R]
reconst_tci = [tci(QD.grididx_to_quantics(grid, i)) for i in 1:2^R]
reconst_tci_globalpivot = [tci_globalpivot(QD.grididx_to_quantics(grid, i)) for i in 1:2^R]

#==
fig, ax = plt.subplots()
ax.plot(ref, label="ref", marker="", linestyle="--")
ax.plot(reconst_tci, label="TCI without global pivot", marker="x", linestyle="")
ax.plot(reconst_tci_globalpivot, label="TCI with global pivot", marker="+", linestyle="")
ax.set_title("Adding global pivot")
ax.set_xlabel("Index")
ax.legend()
fig.savefig("global_pivot.pdf")
==#

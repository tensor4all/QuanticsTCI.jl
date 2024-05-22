using QuanticsTCI
import QuanticsGrids as QG

R = 40                                      # Number of bits <@$\cR$@>
M = 2^R                                     # Number of discretization points <@$M$@>
xgrid = QG.DiscretizedGrid{1}(R, -10, 10)   # Discretization grid <@$x(\bsigma)$@>

function f(x)                               # Function of interest <@$f(x)$@>
    return (
        sinc(x) + 3 * exp(-0.3 * (x - 4)^2) * sinc(x - 4) - cos(4 * x)^2 -
        2 * sinc(x + 10) * exp(-0.6 * (x + 9)) + 4 * cos(2 * x) * exp(-abs(x + 5)) +
        6 * 1 / (x - 11) + sqrt(abs(x)) * atan(x / 15))
end

# Construct and optimize quantics TCI <@$\tf_\bsigma$@>
f_tci, ranks, errors = quanticscrossinterpolate(Float64, f, xgrid; maxbonddim=12)
# Print a table to compare <@$f(x)$@> and <@$\tF_\bsigma$@> on some regularly spaced points
println("x\t f(x)\t\t\t f_tt(x)")
for m in 1:2^(R-5):M
    x = QG.grididx_to_origcoord(xgrid, m)
    println("$x\t$(f(x))\t$(f_tci(m))")
end

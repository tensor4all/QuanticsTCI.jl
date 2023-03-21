function displayhamiltonian(
    ax,
    H::Matrix{Float64},
    lattice::TCI.IndexSet;
    vmax::Float64=maximum(abs.(H)),
    cm=get_cmap("Purples")
)
    ax.set_aspect(1)

    for s in lattice.fromint
        for n in neighbours(s)
            rs, rn = realspacecoordinates.([s, n])
            ax.plot([rs[1], rn[1]], [rs[2], rn[2]], color="gray", linewidth=0.5)
        end
    end

    coords = realspacecoordinates.(lattice.fromint)
    for (i, ri) in enumerate(coords)
        for (j, rj) in enumerate(coords)
            value = H[i, j] / vmax
            ax.plot(
                [ri[1], rj[1]], [ri[2], rj[2]],
                color=cm(value),
                zorder=1 + value,
                alpha=0.5)
        end
    end
end

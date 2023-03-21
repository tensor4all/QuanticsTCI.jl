"""
    qtt_to_mps(qtt, siteindices...)

Convert a quantics tensor train to an ITensor MPS

 * `qtt`            Tensor train as an array of tensors
 * `siteindices`    Arrays of ITensor Index objects
"""
function qtt_to_mps(qtt, siteindices...)
    n = length(qtt)

    if isempty(siteindices)
        siteindices = [[Index(size(t, 2), "site") for t in qtt]]
    end

    qttmps = MPS(n)
    links =
        [
            Index(1, "link"),
            [Index(size(t, 3), "link") for t in qtt]...
        ]

    qttmps[1] = ITensor(
        qtt[1],
        [s[1] for s in siteindices]...,
        links[2]
    )
    for i in 2:(n-1)
        qttmps[i] = ITensor(
            qtt[i],
            links[i],
            [s[i] for s in siteindices]...,
            links[i+1]
        )
    end
    qttmps[n] = ITensor(
        qtt[n],
        links[n],
        [s[n] for s in siteindices]...
    )

    return qttmps
end

using QuanticsTCI
using Documenter

DocMeta.setdocmeta!(QuanticsTCI, :DocTestSetup, :(using QuanticsTCI); recursive=true)

makedocs(;
    modules=[QuanticsTCI],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    sitename="QuanticsTCI.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/QuanticsTCI.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md",
        "API Reference" => "apireference.md",
    ])

deploydocs(;
    repo="github.com/tensor4all/QuanticsTCI.jl.git",
    devbranch="main",
)

using QuanticsTCI
using Documenter

DocMeta.setdocmeta!(QuanticsTCI, :DocTestSetup, :(using QuanticsTCI); recursive=true)

makedocs(;
    modules=[QuanticsTCI],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    repo="https://github.com/tensor4all/QuanticsTCI.jl/blob/{commit}{path}#{line}",
    sitename="QuanticsTCI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        repolink="https://github.com/tensor4all/QuanticsTCI.jl",
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

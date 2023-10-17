using QuanticsTCI
using Documenter

DocMeta.setdocmeta!(QuanticsTCI, :DocTestSetup, :(using QuanticsTCI); recursive=true)

makedocs(;
    modules=[QuanticsTCI],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    repo="https://gitlab.com/tensors4fields/QuanticsTCI.jl/blob/{commit}{path}#{line}",
    sitename="QuanticsTCI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        repolink="https://gitlab.com/tensors4fields/QuanticsTCI.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md",
        "API Reference" => "apireference.md",
    ])

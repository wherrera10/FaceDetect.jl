using Documenter, FaceDetect

DocMeta.setdocmeta!(FaceDetect, :DocTestSetup, :(using FaceDetect); recursive=true)

makedocs(;
    modules=[FaceDetect],
    authors="William Herrera",
    sitename="FaceDetect.jl Documentation",
    repo=Documenter.Remotes.GitHub("wherrera10", "FaceDetect.jl"),
    format=Documenter.HTML(;
        canonical="https://wherrera10.github.io/FaceDetect.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        # Add future pages here, e.g., "API Reference" => "api.md"
    ],
)

if get(ENV, "GITHUB_WORKFLOW", "") != "CI"
    deploydocs(;
        repo="github.com/wherrera10/FaceDetect.jl.git",
        devbranch="main",
    )
end

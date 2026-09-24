using Documenter, FaceDetect

DocMeta.setdocmeta!(FaceDetect, :DocTestSetup, :(using OpenCVFaceDetection); recursive=true)

makedocs(;
    modules=[OpenCVFaceDetection],
    authors="William Herrera",
    sitename="OpenCVFaceDetection.jl Documentation",
    repo=Documenter.Remotes.GitHub("wherrera10", "OpenCVFaceDetection.jl"),
    format=Documenter.HTML(;
        canonical="https://wherrera10.github.io/OpenCVFaceDetection.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        # Add future pages here, e.g., "API Reference" => "api.md"
    ],
)

if get(ENV, "FACEDETECT_DEPLOY_DOCS", "true") == "true"
    deploydocs(;
        repo="github.com/wherrera10/OpenCVFaceDetection.jl.git",
        devbranch="main",
    )
end

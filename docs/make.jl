using Documenter
using RunwayLib

makedocs(
    sitename="RunwayLib",
    format=Documenter.HTML(),
    modules=[RunwayLib],
    repo=Remotes.GitHub("RomeoV", "RunwayLib.jl"),
    pages=[
        "RunwayLib.jl: Fast Pose Estimation and Runtime Assurance for Runway Landings." => "index.md",
        "getting_started.md",
        "camera_model.md",
        "python_interface.md",
        "C_interface.md",
        "noise_models.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/RomeoV/RunwayLib.jl.git"
)

using Documenter
using RunwayLib

makedocs(
    sitename="RunwayLib",
    format=Documenter.HTML(),
    modules=[RunwayLib],
    repo=Remotes.GitHub("RomeoV", "RunwayLib.jl"),
    pages=[
        "RunwayLib.jl: Fast Pose Estimation and Runtime Assurance for Runway Landings." => "index.md",
        "Getting started" => "getting_started.md",
        # Camera Models
        # Noise Models
        # C Interface
        # Julia interface
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
